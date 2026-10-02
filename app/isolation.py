"""
停管隔离作业 —— 可逐步执行的停管/复管操作（仅存于单进程内存，重启即清空）。

每个作业记录：创建时校验通过的管网（沿用 solver 的校验与稳态求解）、允许暂时
失供的用水节点集合（可为空）、当前关闭管段集合、版本号与最近一次成功计算的
状态快照。管段初始全部开启。

每次“切换一根现有管段的开闭”都按全部关闭管段重新计算水源连通区：

  * 失供节点（与水源断开的用水节点）必须全部在允许暂时失供的集合内；
  * 连通区内用水节点按原需水量求解，且任何用水节点水头不得低于最低限值；
  * 失供节点保留原需水量，重新连通后自动恢复参与求解；
  * 连通区仅剩水源时供水量为零（不再调用求解器）。

只有全部规则满足且稳态求解收敛时才一次性提交：关闭集合、版本号 +1、状态快照；
非法操作、不收敛或规则不满足时全部保持原状并说明原因。版本号即乐观锁：
同一版本的并发切换最多成功一次；读取始终返回某一版本的完整快照，绝不混合
不同版本的数据。

作业执行中还可以在**不重建作业**的前提下修订用水节点需水量：提交作业 ID、
当前版本与非空修订列表（每项含现有用水节点编号 node_id 与非负有限的新
需水量 demand，列表内节点互异、布尔值一律拒绝）。后端只改列出的节点
（水源、管段、最低水头限值与允许失供集合均不变），并按**当前关闭集合**
重算：连通节点使用新需水量求解且不得欠压；失供节点只保存新值并更新失供
总量（不参与求解），重新连通后自然按新值参与求解。只有规则满足且收敛时
才一次性提交：新需水量、完整快照与版本号 +1；非法输入、欠压或不收敛均
不改变状态。修订与切换共用同一把版本乐观锁：同一版本的修订与切换合计
最多成功一次，落后提交返回 409 与 current_version，基于旧需水量算成的
结果绝不允许覆盖新需水量。

耗时的连通区重算与稳态求解在作业锁之外进行，因此计算期间健康检查、已提交
快照的读取以及其他作业的操作都不会被阻塞；同一作业也允许后来的切换或修订
先算完并先提交。临界区只在提交瞬间持有：提交时重新比对版本，版本已被后来者
推进则本次结果作废、返回 409 与当前版本，绝不覆盖新状态；规则校验或求解
失败同样不推进版本。

模块级 ``_COMPUTE_GATE`` 仅供测试使用：生产环境为 None；测试可把它替换成
可调用对象，在每次重算开始（规则判定与求解器调用之前）同步阻塞该次计算，
从而用“暂停/放行”而不是大管网或长时间等待来确定 HTTP 并发顺序。stage
依次为 ``"create"`` / ``"toggle"`` / ``"revise"``（创建 / 切换管段 /
修订需水量）。
"""

from __future__ import annotations

import math
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
            "同一版本的修订与切换合计最多成功一次，请重新读取最新状态后再试。")
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
    demands: dict[str, float]  # 用水节点当前需水量（创建后可经修订更新；按录入顺序消费）
    version: int
    snapshot: dict[str, Any]
    lock: threading.Lock = field(default_factory=threading.Lock)


# 作业注册表：仅存于单进程内存，服务重启即清空
_JOBS: dict[str, _IsolationJob] = {}
_REGISTRY_LOCK = threading.Lock()

# 测试钩子：None（生产默认）表示直通；测试可替换为形如
# gate(job_id, stage, *, version, closed) -> None 的可调用对象，
# 在每次快照重算开始时同步调用并允许阻塞（作业锁在此时并不持有）。
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


def _validate_demand_changes(raw: Any, demand_ids: set[str]) -> dict[str, float]:
    """校验需水量修订列表：非空数组，每项含互异现有用水节点 node_id 与
    非负有限 demand（布尔值拒绝）。返回 {node_id: 新需水量}（按请求出现顺序）。
    """
    if not isinstance(raw, list):
        raise ValidationError(
            "changes 必须是非空的修订项数组，每项包含 node_id 与 demand。")
    if not raw:
        raise ValidationError(
            "changes 不能为空：至少给出一个要修订的用水节点需水量。")
    changes: dict[str, float] = {}
    for k, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValidationError(f"第 {k} 个修订项必须是包含 node_id 与 demand 的对象。")
        nid = item.get("node_id")
        if not isinstance(nid, str) or not nid.strip():
            raise ValidationError(
                f"第 {k} 个修订项缺少有效的 node_id（非空字符串）。")
        nid = nid.strip()
        if nid not in demand_ids:
            raise ValidationError(
                f"修订节点“{nid}”不是该作业内已定义的用水节点"
                "（只能修订现有用水节点，不能是水源或未定义节点）。")
        if nid in changes:
            raise ValidationError(
                f"修订节点“{nid}”在列表中重复，修订项节点必须互异。")
        value = item.get("demand")
        # bool 是 int 的子类：必须先于数值判定显式拒绝
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(
                f"用水节点“{nid}”的新需水量必须是非负有限数值。")
        demand = float(value)
        if not math.isfinite(demand):
            raise ValidationError(
                f"用水节点“{nid}”的新需水量必须是有限数值"
                "（不允许 NaN 或无穷大）。")
        if demand < 0.0:
            raise ValidationError(
                f"用水节点“{nid}”的新需水量不能为负值，当前为 {demand:g}。")
        changes[nid] = demand
    return changes


def _network_with_demands(net: Network,
                          demands: dict[str, float]) -> Network:
    """用当前需水量映射替换 Network 内用水节点的 demand（拓扑与限值不变）。"""
    nodes: list[dict[str, Any]] = []
    for node in net.nodes:
        if node["type"] == "demand":
            updated = dict(node)
            updated["demand"] = float(demands[node["id"]])
            nodes.append(updated)
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
                    demands: dict[str, float],
                    allowed: frozenset[str], closed: frozenset[str],
                    *, action: str) -> dict[str, Any]:
    """计算某一关闭集合与当前需水量对应的状态快照；规则不满足或不收敛时
    抛 IsolationRuleError。

    action 决定失败文案与求解门 stage：
      * ``"create"`` —— 创建时的初始状态（全部管段开启），失败文案为“不创建作业”；
      * ``"toggle"`` —— 切换管段，失败文案为“本次操作未生效，作业保持原状”；
      * ``"revise"`` —— 修订需水量（关闭集合不变），失败文案与切换同理。
    """
    if _COMPUTE_GATE is not None:
        _COMPUTE_GATE(job_id, action,
                      version=version, closed=frozenset(closed))

    # 当前需水量可能已被修订：求解一律走替换过 demand 的工作网络；
    # 拓扑（节点/管段编号与顺序）仍取自原始 network。
    work_net = _network_with_demands(net, demands)
    source_id = next(n["id"] for n in net.nodes if n["type"] == "source")
    demand_map = {n["id"]: n["demand"] for n in work_net.nodes
                  if n["type"] == "demand"}

    component_nodes, component_pipes = _connected_component(net, closed)
    component_set = set(component_nodes)
    # 失供节点保存当前需水量（重连后自动按新值参与求解）
    unserved = [nid for nid in demand_map if nid not in component_set]

    # 规则 1：失供节点必须全部在允许暂时失供的集合内
    outside = [nid for nid in unserved if nid not in allowed]
    if outside:
        names = "、".join(f"“{nid}”" for nid in outside)
        if action == "create":
            raise IsolationRuleError(
                f"初始状态下用水节点 {names} 即失供，且不在允许暂时失供的集合内，"
                "不创建停管隔离作业。")
        prefix = "修订需水量不会改变管段开闭，但" if action == "revise" else "切换后"
        raise IsolationRuleError(
            f"{prefix}用水节点 {names} 将失供，但其不在允许暂时失供的节点集合内；"
            "本次操作未生效，关闭集合、需水量、版本与稳态结果全部保持原状。")

    # 规则 2：连通区用水节点按当前需水量求解且不得欠压；仅剩水源时供水量为零
    if len(component_nodes) == 1:
        solution = _source_only_solution(
            net, next(n for n in net.nodes if n["type"] == "source"))
    else:
        sub_net = Network(
            nodes=[n for n in work_net.nodes if n["id"] in component_set],
            pipes=component_pipes,
            min_head=net.min_head,
        )
        try:
            solution = solve_network(sub_net)
        except SolverNotConverged as exc:
            if action == "create":
                raise IsolationRuleError(
                    "初始状态（全部管段开启）的稳态求解未能收敛，不创建停管隔离作业。"
                    f"（{exc}）") from exc
            if action == "revise":
                raise IsolationRuleError(
                    "按新需水量重算后连通区的稳态求解未能收敛；本次修订未生效，"
                    f"需水量、关闭集合、版本与稳态结果全部保持原状。（{exc}）") from exc
            raise IsolationRuleError(
                "切换后连通区的稳态求解未能收敛；本次操作未生效，关闭集合、版本与"
                f"稳态结果全部保持原状。（{exc}）") from exc
        below = [nid for nid in component_nodes
                 if nid in demand_map and solution["heads"][nid] < net.min_head]
        if below:
            names = "、".join(f"“{nid}”" for nid in below)
            if action == "create":
                raise IsolationRuleError(
                    f"初始状态（全部管段开启）下，用水节点 {names} 的水头低于最低"
                    f"水头限值 {net.min_head:g}，不创建停管隔离作业。")
            if action == "revise":
                raise IsolationRuleError(
                    f"按新需水量重算后，连通区内用水节点 {names} 的水头将低于最低"
                    f"水头限值 {net.min_head:g}；本次修订未生效，需水量、关闭集合、"
                    "版本与稳态结果全部保持原状。")
            raise IsolationRuleError(
                f"切换后连通区内用水节点 {names} 的水头将低于最低水头限值 "
                f"{net.min_head:g}；本次操作未生效，关闭集合、版本与稳态结果全部保持原状。")

    return {
        "job_id": job_id,
        "version": version,
        "min_head": net.min_head,
        "source_node": source_id,
        "allowed_unserved": [nid for nid in demand_map if nid in allowed],
        # 全部当前需水量（含已修订值），按节点录入顺序；修订/切换/读取都带
        "demands": [{"id": nid, "demand": float(demand_map[nid])}
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
# 作业生命周期：创建 / 读取 / 切换
# --------------------------------------------------------------------------- #
def create_job(payload: Any) -> dict[str, Any]:
    """创建停管隔离作业，返回版本 0 的初始快照。

    沿用同一套管网校验与稳态求解；校验失败、初始有用水节点欠压或初始求解
    不收敛时抛异常，不创建作业。初始重算不持有任何作业锁，阻塞期间不影响
    其他作业的读取与切换。
    """
    net = validate_payload(payload)
    allowed = _validate_allow_unserved(payload.get("allow_unserved"), net)
    job_id = uuid.uuid4().hex
    demands = {n["id"]: float(n["demand"]) for n in net.nodes
               if n["type"] == "demand"}
    # 重算在注册表锁之外完成；通过前作业尚不存在，失败自然不会留下任何状态
    snapshot = _build_snapshot(job_id, 0, net, demands, allowed,
                               frozenset(), action="create")
    job = _IsolationJob(
        job_id=job_id,
        network=net,
        allowed_unserved=allowed,
        closed_pipes=frozenset(),
        demands=demands,
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

    版本不一致抛 VersionConflictError（同一版本的并发切换最多成功一次）；
    管段不存在、规则不满足或求解不收敛时抛异常，关闭集合、版本号与稳态
    结果全部保持原状。

    耗时的快照重算在作业锁之外进行：同一作业的多个切换可以同时计算，
    后来的切换甚至可以先算完先提交。临界区只在提交瞬间出现：提交时重新
    比对版本，若在本次计算期间版本已被其他成功切换推进，则本次结果作废
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
        base_demands = dict(job.demands)
        net = job.network
        allowed = job.allowed_unserved

    new_closed = base_closed ^ {pipe_id}  # 开→关 或 关→开
    # 锁外重算：其他请求（健康检查、读取、其他作业、后来的切换）在此期间照常返回
    candidate = _build_snapshot(job.job_id, base_version + 1, net,
                                base_demands, allowed, new_closed,
                                action="toggle")

    # 第 2 段临界区：提交时重新比对版本——计算期间版本若已推进则丢弃本次结果
    with job.lock:
        if job.version != base_version:
            raise VersionConflictError(version, job.version)
        # 全部规则满足且求解收敛：一次性提交关闭集合、版本号与稳态结果
        job.closed_pipes = new_closed
        job.version = base_version + 1
        job.snapshot = candidate
        return candidate


def revise_demands(job_id: str, payload: Any) -> tuple[dict[str, Any], list[str]]:
    """携带当前版本修订一个或多个现有用水节点的需水量，返回（新版本快照，
    本次修订生效的节点编号列表，按请求顺序）。

    只改修订列表中列出的节点；水源、管段、最低水头限值与允许失供集合均不变，
    关闭集合也不变，按当前关闭集合与新需水量整体重算：连通节点使用新值且不得
    欠压，失供节点只保存新值并更新失供总量，重连后自然按新值参与求解。

    版本不一致抛 VersionConflictError（同一版本的修订与切换合计最多成功一次）；
    修订列表非法、欠压或不收敛时抛异常，需水量、关闭集合、版本号与稳态结果
    全部保持原状。锁外重算与提交时版本比对的并发语义与 toggle_pipe 完全一致：
    计算期间版本若已被别的切换/修订推进，本次结果作废并抛 VersionConflictError
    （携带作业当前版本），基于旧需水量算成的结果绝不覆盖新需水量。
    """
    with _REGISTRY_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise JobNotFoundError(job_id)
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是包含 version 与 changes 的对象。")
    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise ValidationError("version 必须是非负整数（即作业当前版本号）。")
    with job.lock:
        demand_ids = {n["id"] for n in job.network.nodes
                      if n["type"] == "demand"}
    changes = _validate_demand_changes(payload.get("changes"), demand_ids)

    # 第 1 段临界区：核对版本并取基态（关闭集合与当前需水量），立即释放
    with job.lock:
        if version != job.version:
            raise VersionConflictError(version, job.version)
        base_version = job.version
        base_closed = job.closed_pipes
        new_demands = dict(job.demands)
        net = job.network
        allowed = job.allowed_unserved
    new_demands.update(changes)  # 只改列出的节点，其余保持当前值

    # 锁外重算：关闭集合不变，仅需水量可能改变
    candidate = _build_snapshot(job.job_id, base_version + 1, net,
                                new_demands, allowed, base_closed,
                                action="revise")

    # 第 2 段临界区：提交时重新比对版本——旧计算不得覆盖新需水量/新关闭集合
    with job.lock:
        if job.version != base_version:
            raise VersionConflictError(version, job.version)
        # 全部规则满足且求解收敛：一次性提交需水量、完整快照与版本号
        job.demands = new_demands
        job.version = base_version + 1
        job.snapshot = candidate
        return candidate, list(changes)
