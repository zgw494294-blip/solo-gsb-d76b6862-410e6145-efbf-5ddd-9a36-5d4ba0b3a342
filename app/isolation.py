"""
停管隔离作业 —— 可逐步执行的停管/复管操作与作业内需水量修订（仅存于单进程
内存，重启即清空）。

每个作业记录：创建时校验通过的管网（沿用 solver 的校验与稳态求解；需水量可在
作业执行中修订，水源、管段、最低水头与允许失供集合永不改变）、允许暂时失供的
用水节点集合（可为空）、当前关闭管段集合、版本号与最近一次成功计算的状态快照。
管段初始全部开启。

每次“切换一根现有管段的开闭”或“修订一批用水节点的需水量”，都按全部关闭
管段重新计算水源连通区：

  * 失供节点（与水源断开的用水节点）必须全部在允许暂时失供的集合内；
  * 连通区内用水节点按当前需水量（含已提交的修订）求解，且任何用水节点水头
    不得低于最低限值；
  * 失供节点也保存新需水量并计入失供总量，重新连通后自动按新值参与求解；
  * 连通区仅剩水源时供水量为零（不再调用求解器）。

只有全部规则满足且稳态求解收敛时才一次性提交：切换提交关闭集合、修订提交
需水量，二者都提交完整快照并把版本号 +1；非法操作、不收敛或规则不满足时
全部保持原状并说明原因。版本号即乐观锁：同一版本的修订与管段切换合计最多
成功一次；读取始终返回某一版本的完整快照，绝不混合不同版本的数据。

耗时的连通区重算与稳态求解在作业锁之外进行，因此计算期间健康检查、已提交
快照的读取以及其他作业的操作都不会被阻塞；同一作业也允许后来的切换先算完
并先提交。临界区只在提交瞬间持有：提交时重新比对版本，版本已被后来者推进
则本次结果作废、返回 409 与当前版本，绝不覆盖新状态；规则校验或求解失败
同样不推进版本。

模块级 ``_COMPUTE_GATE`` 仅供测试使用：生产环境为 None；测试可把它替换成
可调用对象，在每次重算开始（规则判定与求解器调用之前）同步阻塞该次计算，
从而用“暂停/放行”而不是大管网或长时间等待来确定 HTTP 并发顺序。
``stage`` 取 ``"create"`` / ``"toggle"`` / ``"revise"``，分别对应创建、
管段切换与需水量修订触发的重算。
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .solver import (
    CHECK_TOL,
    Network,
    SolverNotConverged,
    ValidationError,
    _as_finite,
    solve_network,
    validate_payload,
)


class JobNotFoundError(KeyError):
    """作业编号不存在（作业仅存于单进程内存，服务重启后即清空）。"""


class VersionConflictError(RuntimeError):
    """请求携带的版本与作业当前版本不一致：同一版本最多允许一次成功提交。"""

    def __init__(self, expected: int, current: int) -> None:
        super().__init__(
            f"版本冲突：请求携带的版本为 {expected}，作业当前版本已为 {current}；"
            "同一版本的修订与管段切换合计最多成功一次，"
            "请重新读取最新状态后再试。")
        self.expected = expected
        self.current = current


class IsolationRuleError(RuntimeError):
    """切换后不满足停管隔离规则（或连通区求解不收敛）：作业状态全部保持原状。"""


@dataclass
class _IsolationJob:
    """单个停管隔离作业的全部可变状态（仅在持有自身锁时修改）。"""
    job_id: str
    network: Network
    allowed_unserved: frozenset[str]
    closed_pipes: frozenset[str]
    version: int
    snapshot: dict[str, Any]
    lock: threading.Lock = field(default_factory=threading.Lock)


# 作业注册表：仅存于单进程内存，服务重启即清空
_JOBS: dict[str, _IsolationJob] = {}
_REGISTRY_LOCK = threading.Lock()

# 测试钩子：None（生产默认）表示直通；测试可替换为形如
# gate(job_id, stage, *, version, closed) -> None 的可调用对象，
# 在每次快照重算开始时同步调用并允许阻塞（作业锁在此时并不持有）。
# stage 为 "create" / "toggle" / "revise" 之一。
_COMPUTE_GATE: Callable[..., None] | None = None


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _validate_allow_unserved(raw: Any, net: Network) -> frozenset[str]:
    """校验允许暂时失供的用水节点集合（可省略或为空数组），返回 frozenset。"""
    if raw is None:
        return frozenset()
    if not isinstance(raw, list):
        raise ValidationError(
            "allow_unserved 必须是允许暂时失供的用水节点编号数组（可为空数组）。")
    demand_ids = {n["id"] for n in net.nodes if n["type"] == "demand"}
    seen: set[str] = set()
    for k, item in enumerate(raw, start=1):
        if not isinstance(item, str) or not item.strip():
            raise ValidationError(
                f"允许失供集合第 {k} 项缺少有效的节点编号（非空字符串）。")
        nid = item.strip()
        if nid not in demand_ids:
            raise ValidationError(
                f"允许失供节点“{nid}”不是已定义的用水节点"
                "（集合内必须全部为用水节点，不能是水源或未定义节点）。")
        if nid in seen:
            raise ValidationError(f"允许失供节点“{nid}”重复，集合内节点必须互异。")
        seen.add(nid)
    return frozenset(seen)


def _validate_demand_updates(raw: Any, net: Network) -> list[dict[str, float]]:
    """校验需水量修订列表，返回按请求顺序的 [{"node_id","demand"}]。

    列表必须非空；每项含 node_id（作业内互异的现有用水节点编号）与 demand
    （非负有限数，拒绝布尔值）。水源、未知节点、重复节点一律拒绝。
    """
    if not isinstance(raw, list):
        raise ValidationError(
            "demands 必须是非空的修订项数组，每项包含 node_id 与 demand。")
    if not raw:
        raise ValidationError(
            "demands 修订列表不能为空：至少需要一项 node_id 与 demand。")
    demand_ids = {n["id"] for n in net.nodes if n["type"] == "demand"}
    updates: list[dict[str, float]] = []
    seen: set[str] = set()
    for k, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValidationError(f"修订列表第 {k} 项必须是包含 node_id 与 demand 的对象。")
        nid = item.get("node_id")
        if not isinstance(nid, str) or not nid.strip():
            raise ValidationError(
                f"修订列表第 {k} 项缺少有效的 node_id（非空字符串）。")
        nid = nid.strip()
        if nid not in demand_ids:
            raise ValidationError(
                f"修订节点“{nid}”不是该作业中已定义的用水节点"
                "（只能修订现有用水节点的需水量，不能是水源或未定义节点）。")
        if nid in seen:
            raise ValidationError(
                f"修订节点“{nid}”在列表中重复，一次修订内的节点必须互异。")
        demand = _as_finite(
            item.get("demand"), f"用水节点“{nid}”的修订需水量", nonneg=True)
        seen.add(nid)
        updates.append({"node_id": nid, "demand": demand})
    return updates


def _network_with_demands(net: Network, demand_map: dict[str, float]) -> Network:
    """按 demand_map（用水节点编号 -> 当前需水量）构造新 Network。

    水源、管段与最低水头保持不变；节点与管段的录入顺序也保持不变，
    因此快照中一切“按录入顺序”的字段在修订后仍稳定。
    """
    nodes: list[dict[str, Any]] = []
    for node in net.nodes:
        if node["type"] == "demand":
            nodes.append({"id": node["id"], "type": "demand",
                          "demand": float(demand_map[node["id"]])})
        else:
            nodes.append(node)
    return Network(nodes=nodes, pipes=net.pipes, min_head=net.min_head)


# --------------------------------------------------------------------------- #
# 状态计算
# --------------------------------------------------------------------------- #
def _connected_component(net: Network, closed: frozenset[str]):
    """按全部关闭管段做无向 BFS，返回（连通区节点编号、连通区内开启管段），均按录入顺序。"""
    source_id = next(n["id"] for n in net.nodes if n["type"] == "source")
    adj: dict[str, list[str]] = {n["id"]: [] for n in net.nodes}
    for pipe in net.pipes:
        if pipe["id"] in closed:
            continue
        adj[pipe["start"]].append(pipe["end"])
        adj[pipe["end"]].append(pipe["start"])
    seen = {source_id}
    queue = deque([source_id])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    nodes = [n["id"] for n in net.nodes if n["id"] in seen]
    # 开启管段的一个端点可达，另一个端点必然也可达
    pipes = [p for p in net.pipes if p["id"] not in closed and p["start"] in seen]
    return nodes, pipes


def _source_only_solution(net: Network, source: dict[str, Any]) -> dict[str, Any]:
    """连通区仅剩水源时的平凡解：不调用求解器，供水量为零，键与 solve_network 一致。"""
    head = float(source["head"])
    below = head < net.min_head
    return {
        "converged": True,
        "tolerance": CHECK_TOL,
        "method": "source-only",
        "iterations": 0,
        "solver_residual_inf": 0.0,
        "node_results": [{
            "id": source["id"],
            "type": "source",
            "head": head,
            "net_inflow": 0.0,
            "below_min_head": below,
            "fixed_head": head,
            "outflow": 0.0,
        }],
        "pipe_results": [],
        "heads": {source["id"]: head},
        "flows": {},
        "source_node": source["id"],
        "source_supply": 0.0,
        "total_demand": 0.0,
        "balance_residual": 0.0,
        "max_node_residual": 0.0,
        "max_pipe_residual": 0.0,
        "min_head": net.min_head,
        "below_min_head": [source["id"]] if below else [],
    }


def _build_snapshot(job_id: str, version: int, net: Network,
                    allowed: frozenset[str], closed: frozenset[str],
                    *, stage: str) -> dict[str, Any]:
    """计算某一关闭集合、某套当前需水量对应的状态快照。

    规则不满足或不收敛时抛 IsolationRuleError。``stage`` 取 "create"（创建
    初始状态，失败文案为“不创建作业”）、"toggle"（管段切换）或 "revise"
    （需水量修订）；后两者失败文案均为“本次操作未生效，作业保持原状”。
    """
    if _COMPUTE_GATE is not None:
        _COMPUTE_GATE(job_id, stage, version=version, closed=frozenset(closed))

    source_id = next(n["id"] for n in net.nodes if n["type"] == "source")
    demand_map = {n["id"]: n["demand"] for n in net.nodes if n["type"] == "demand"}

    component_nodes, component_pipes = _connected_component(net, closed)
    component_set = set(component_nodes)
    # 失供节点同样保存当前需水量，重连后自动按新值参与求解
    unserved = [nid for nid in demand_map if nid not in component_set]

    # 规则 1：失供节点必须全部在允许暂时失供的集合内
    outside = [nid for nid in unserved if nid not in allowed]
    if outside:
        names = "、".join(f"“{nid}”" for nid in outside)
        if stage == "create":
            raise IsolationRuleError(
                f"初始状态下用水节点 {names} 即失供，且不在允许暂时失供的集合内，"
                "不创建停管隔离作业。")
        action = "修订需水量" if stage == "revise" else "切换"
        raise IsolationRuleError(
            f"{action}后用水节点 {names} 将失供，但其不在允许暂时失供的节点集合内；"
            "本次操作未生效，关闭集合、版本与稳态结果全部保持原状。")

    # 规则 2：连通区用水节点按当前需水量求解且不得欠压；仅剩水源时供水量为零
    if len(component_nodes) == 1:
        solution = _source_only_solution(
            net, next(n for n in net.nodes if n["type"] == "source"))
    else:
        sub_net = Network(
            nodes=[n for n in net.nodes if n["id"] in component_set],
            pipes=component_pipes,
            min_head=net.min_head,
        )
        try:
            solution = solve_network(sub_net)
        except SolverNotConverged as exc:
            if stage == "create":
                raise IsolationRuleError(
                    "初始状态（全部管段开启）的稳态求解未能收敛，不创建停管隔离作业。"
                    f"（{exc}）") from exc
            if stage == "revise":
                raise IsolationRuleError(
                    "修订需水量后连通区的稳态求解未能收敛；本次操作未生效，需水量、"
                    f"关闭集合、版本与稳态结果全部保持原状。（{exc}）") from exc
            raise IsolationRuleError(
                "切换后连通区的稳态求解未能收敛；本次操作未生效，关闭集合、版本与"
                f"稳态结果全部保持原状。（{exc}）") from exc
        below = [nid for nid in component_nodes
                 if nid in demand_map and solution["heads"][nid] < net.min_head]
        if below:
            names = "、".join(f"“{nid}”" for nid in below)
            if stage == "create":
                raise IsolationRuleError(
                    f"初始状态（全部管段开启）下，用水节点 {names} 的水头低于最低"
                    f"水头限值 {net.min_head:g}，不创建停管隔离作业。")
            if stage == "revise":
                raise IsolationRuleError(
                    f"修订需水量后连通区内用水节点 {names} 的水头将低于最低水头限值 "
                    f"{net.min_head:g}；本次操作未生效，需水量、关闭集合、版本与稳态"
                    "结果全部保持原状。")
            raise IsolationRuleError(
                f"切换后连通区内用水节点 {names} 的水头将低于最低水头限值 "
                f"{net.min_head:g}；本次操作未生效，关闭集合、版本与稳态结果全部保持原状。")

    return {
        "job_id": job_id,
        "version": version,
        "min_head": net.min_head,
        "source_node": source_id,
        "allowed_unserved": [nid for nid in demand_map if nid in allowed],
        # 全部当前需水量（含失供节点保存的新值），按节点录入顺序
        "demands": [{"node_id": nid, "demand": float(demand_map[nid])}
                    for nid in demand_map],
        "closed_pipes": [p["id"] for p in net.pipes if p["id"] in closed],
        "open_pipes": [p["id"] for p in net.pipes if p["id"] not in closed],
        "pipe_states": [
            {"id": p["id"], "start": p["start"], "end": p["end"],
             "closed": p["id"] in closed}
            for p in net.pipes
        ],
        "unserved_nodes": unserved,
        "unserved_demand": float(sum(demand_map[nid] for nid in unserved)),
        "component_nodes": component_nodes,
        "component_pipes": [p["id"] for p in component_pipes],
        "source_only": len(component_nodes) == 1,
        "source_supply": solution["source_supply"],
        "total_demand": solution["total_demand"],
        "solution": solution,
    }


# --------------------------------------------------------------------------- #
# 作业生命周期：创建 / 读取 / 切换 / 修订需水量
# --------------------------------------------------------------------------- #
def create_job(payload: Any) -> dict[str, Any]:
    """创建停管隔离作业，返回版本 0 的初始快照。

    沿用同一套管网校验与稳态求解；校验失败、初始有用水节点欠压或初始求解
    不收敛时抛异常，不创建作业。初始重算不持有任何作业锁，阻塞期间不影响
    其他作业的读取与操作。
    """
    net = validate_payload(payload)
    allowed = _validate_allow_unserved(payload.get("allow_unserved"), net)
    job_id = uuid.uuid4().hex
    # 重算在注册表锁之外完成；通过前作业尚不存在，失败自然不会留下任何状态
    snapshot = _build_snapshot(job_id, 0, net, allowed, frozenset(), stage="create")
    job = _IsolationJob(
        job_id=job_id,
        network=net,
        allowed_unserved=allowed,
        closed_pipes=frozenset(),
        version=0,
        snapshot=snapshot,
    )
    with _REGISTRY_LOCK:
        _JOBS[job_id] = job
    return snapshot


def get_snapshot(job_id: str) -> dict[str, Any]:
    """读取作业最近一次成功提交的完整快照（快照为不可变整体，绝不混合不同版本的数据）。

    即使该作业此刻正在重算，本函数也只短暂加锁后立即返回最近已提交的快照，
    不会等待进行中的计算。
    """
    with _REGISTRY_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    with job.lock:
        return job.snapshot


def toggle_pipe(job_id: str, payload: Any) -> dict[str, Any]:
    """携带当前版本切换一根现有管段的开闭，返回新版本快照。

    版本不一致抛 VersionConflictError（同一版本的修订与切换合计最多成功
    一次）；管段不存在、规则不满足或求解不收敛时抛异常，关闭集合、版本号
    与稳态结果全部保持原状。

    耗时的快照重算在作业锁之外进行：同一作业的多个操作可以同时计算，
    后来的操作甚至可以先算完先提交。临界区只在提交瞬间出现：提交时重新
    比对版本，若在本次计算期间版本已被其他成功操作推进，则本次结果作废
    并抛 VersionConflictError（携带作业当前版本），绝不覆盖新状态。
    """
    with _REGISTRY_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是包含 version 与 pipe_id 的对象。")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValidationError("version 必须是非负整数（即作业当前版本号）。")
    pipe_id = payload.get("pipe_id")
    if not isinstance(pipe_id, str) or not pipe_id.strip():
        raise ValidationError("pipe_id 必须是要切换开闭的现有管段编号（非空字符串）。")
    pipe_id = pipe_id.strip()

    # 第 1 段临界区：仅做输入归属核对与基态快照，立即释放，不带着锁做重算
    with job.lock:
        if version != job.version:
            raise VersionConflictError(version, job.version)
        if pipe_id not in {p["id"] for p in job.network.pipes}:
            raise ValidationError(
                f"管段“{pipe_id}”不是该作业管网中已定义的管段。")
        base_version = job.version
        base_closed = job.closed_pipes
        net = job.network
        allowed = job.allowed_unserved

    new_closed = base_closed ^ {pipe_id}  # 开→关 或 关→开
    # 锁外重算：其他请求（健康检查、读取、其他作业、后来的操作）在此期间照常返回
    candidate = _build_snapshot(job.job_id, base_version + 1, net,
                                allowed, new_closed, stage="toggle")

    # 第 2 段临界区：提交时重新比对版本——计算期间版本若已推进则丢弃本次结果
    with job.lock:
        if job.version != base_version:
            raise VersionConflictError(version, job.version)
        # 全部规则满足且求解收敛：一次性提交关闭集合、版本号与稳态结果
        job.closed_pipes = new_closed
        job.version = base_version + 1
        job.snapshot = candidate
        return candidate


def revise_demands(job_id: str, payload: Any) -> tuple[dict[str, Any], list[dict[str, float]]]:
    """携带当前版本修订一批现有用水节点的需水量，返回 (新版本快照, 修订项)。

    修订项为非空列表，每项含 node_id（作业内互异的现有用水节点）与 demand
    （非负有限数，拒绝布尔值）；只改列出的节点，水源、管段、最低水头与允许
    失供集合均不变。按当前关闭集合重算：连通节点使用新需水量且不得欠压，
    失供节点保存新值并更新失供总量（重连后同样按新值求解）。

    版本不一致抛 VersionConflictError（同一版本的修订与管段切换合计最多
    成功一次）；非法输入、欠压或不收敛时抛异常，需水量、关闭集合、版本号
    与稳态结果全部保持原状。锁外重算与提交时版本比对的并发语义与
    ``toggle_pipe`` 完全一致：基于旧版本算成的迟到提交收 409，绝不覆盖
    新需水量。
    """
    with _REGISTRY_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是包含 version 与非空 demands 修订列表的对象。")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValidationError("version 必须是非负整数（即作业当前版本号）。")

    # 第 1 段临界区：版本比对、归属核对与基态快照；修订项校验在此一并完成，
    # 立即释放后不带着锁做重算
    with job.lock:
        if version != job.version:
            raise VersionConflictError(version, job.version)
        updates = _validate_demand_updates(payload.get("demands"), job.network)
        base_version = job.version
        base_closed = job.closed_pipes
        net = job.network
        allowed = job.allowed_unserved

    # 只覆盖列出的节点：在基态网络上构造新网络，其余需水量原样保留
    revised = {n["id"]: n["demand"] for n in net.nodes if n["type"] == "demand"}
    for item in updates:
        revised[item["node_id"]] = item["demand"]
    new_net = _network_with_demands(net, revised)
    # 锁外重算：失供节点保存新值、连通节点按新值求解，均由 _build_snapshot 完成
    candidate = _build_snapshot(job.job_id, base_version + 1, new_net,
                                allowed, base_closed, stage="revise")

    # 第 2 段临界区：提交时重新比对版本——计算期间版本若已推进则丢弃本次结果
    with job.lock:
        if job.version != base_version:
            raise VersionConflictError(version, job.version)
        # 全部规则满足且求解收敛：一次性提交需水量、完整快照与版本号
        job.network = new_net
        job.version = base_version + 1
        job.snapshot = candidate
        return candidate, updates
