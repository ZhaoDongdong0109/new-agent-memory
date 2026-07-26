"""确定性睡眠周期测试

验证情景 -> 语义巩固的完整性质：
1. 相关情景聚类成要点（IDEA 类型，慢衰减），无关情景不受影响
2. 来源归档但可逆：线索仍可唤醒具体情景
3. 要点可检索、可溯源（每行摘要有来源句与分数）
4. 幂等：重复睡眠不产生重复要点
5. 触发器：写入累计重要性达到阈值后 maintain() 自动巩固
6. 图卫生：近零 Hebbian 边被修剪
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HumanLikeMemorySystem
from core.weight_system import MemoryType
from memory_chunk import MemoryLayer


def _make_system(tmp_path, **kwargs):
    kwargs.setdefault("enable_pii_detection", False)
    kwargs.setdefault("enable_audit_log", False)
    return HumanLikeMemorySystem(data_dir=str(tmp_path), **kwargs)


def _add_cluster(system, n=4):
    """一簇相关情景：同人物同主题的多次经历"""
    ids = []
    details = ["聊了项目预算", "确认了交付时间", "讨论了验收标准", "敲定了付款节奏", "复盘了合作流程"]
    for i in range(n):
        cid = system.add_memory(
            content=f"和老王开会{details[i % len(details)]}，进展顺利",
            persons=["老王"],
            topics=["工作", "项目"],
            keywords=["会议", "项目"],
            importance=0.6,
        )
        ids.append(cid)
    return ids


def test_sleep_creates_gist_and_archives_sources(tmp_path):
    system = _make_system(tmp_path)
    cluster_ids = _add_cluster(system, n=4)
    lone_id = system.add_memory(content="独自去公园跑了五公里", topics=["运动"])

    report = system.sleep()

    # 一簇 -> 一条要点
    assert len(report.gists_created) == 1
    gist = system.core.get(report.gists_created[0])
    assert gist is not None
    assert gist.memory_type == MemoryType.IDEA
    assert gist.metadata["source_ids"] == cluster_ids
    assert "经验要点" in gist.content

    # 来源归档（不是删除），并标记归属
    for cid in cluster_ids:
        assert system.core.get(cid) is None
        archived = system.forgotten.get(cid)
        assert archived is not None
        assert archived.metadata["consolidated_into"] == gist.id

    # 无关的孤立记忆不受影响
    assert system.core.get(lone_id) is not None


def test_gist_is_retrievable_and_traceable(tmp_path):
    system = _make_system(tmp_path)
    _add_cluster(system, n=4)
    report = system.sleep()
    gist_id = report.gists_created[0]

    result = system.retrieve("和老王的项目会议")
    assert result.success
    assert gist_id in [c.id for c in result.chunks], "要点未能被常规检索命中"

    # 可溯源：每行摘要句都有分数和来源 id
    gist = system.core.get(gist_id)
    for item in gist.metadata["sentence_scores"]:
        assert item["source_id"] in gist.metadata["source_ids"]
        assert item["score"] > 0


def test_archived_sources_still_wakeable(tmp_path):
    """抽象是可逆的：强线索仍能唤醒具体情景"""
    system = _make_system(tmp_path)
    cluster_ids = _add_cluster(system, n=4)
    system.sleep()

    # 人物 + 主题双锚点唤醒归档的情景
    result = system.retrieve("和老王一起开的项目会议")
    woken = [c.id for c in result.chunks if c.id in cluster_ids]
    assert woken, "归档情景无法被线索唤醒——抽象变成了不可逆的信息销毁"


def test_sleep_is_idempotent(tmp_path):
    system = _make_system(tmp_path)
    _add_cluster(system, n=4)
    first = system.sleep()
    assert len(first.gists_created) == 1

    second = system.sleep()
    assert len(second.gists_created) == 0, "重复睡眠产生了重复要点"


def test_small_clusters_not_abstracted(tmp_path):
    """少于 3 条的相关情景不值得抽象"""
    system = _make_system(tmp_path)
    _add_cluster(system, n=2)
    report = system.sleep()
    assert len(report.gists_created) == 0


def test_maintain_triggers_sleep_at_threshold(tmp_path):
    system = _make_system(tmp_path)
    system.sleep_threshold = 2.0  # 降低阈值便于测试

    _add_cluster(system, n=4)  # 4 * 0.6 = 2.4 >= 2.0
    assert system._importance_since_sleep >= 2.0

    system.maintain()
    assert system._importance_since_sleep == 0.0, "maintain 未触发睡眠"
    gists = [
        c for c in system.core.chunks.values()
        if c.source == "consolidation"
    ]
    assert gists, "触发的睡眠没有产生要点"


def test_graph_hygiene_prunes_weak_edges(tmp_path):
    system = _make_system(tmp_path)
    id_a = system.add_memory(content="记忆甲", topics=["测试"])
    id_b = system.add_memory(content="记忆乙", topics=["测试"])
    chunk_a = system.core.get(id_a)
    chunk_a.associations[id_b] = 0.01  # 近零边
    chunk_a.associations["ghost_1"] = 0.02
    system.core._store.put(chunk_a)

    report = system.sleep()
    assert report.edges_pruned >= 2
    assert not system.core.get(id_a).associations


def test_gist_links_back_to_sources(tmp_path):
    """要点与来源保持 Hebbian 关联：联想回忆可以从要点走回情景"""
    system = _make_system(tmp_path)
    cluster_ids = _add_cluster(system, n=4)
    report = system.sleep()
    gist = system.core.get(report.gists_created[0])
    for cid in cluster_ids:
        assert gist.associations.get(cid, 0) >= 0.5


def test_sources_survive_layer_flag(tmp_path):
    system = _make_system(tmp_path)
    cluster_ids = _add_cluster(system, n=3)
    system.sleep()
    for cid in cluster_ids:
        assert system.forgotten.get(cid).layer == MemoryLayer.FORGOTTEN


# ============ 图式强化：要点支持计数 ============

def test_second_sleep_reinforces_existing_gist(tmp_path):
    """同主题的后续情景增强既有要点，而不是重复抽象出新要点"""
    system = _make_system(tmp_path)
    first_ids = _add_cluster(system, n=3)
    report1 = system.sleep()
    assert len(report1.gists_created) == 1
    gist_id = report1.gists_created[0]

    # 一周后又有 3 次同主题经历
    second_ids = _add_cluster(system, n=3)
    report2 = system.sleep()

    # 不新建，而是强化
    assert report2.gists_created == []
    assert report2.gists_reinforced == [gist_id]

    gist = system.core.get(gist_id)
    assert gist.metadata["support_count"] == 2
    assert gist.metadata["supporting_episodes"] == 6
    assert gist.importance == min(1.0, 0.6 + 0.05 * 6)
    assert "（6次相关经历）" in gist.content
    assert set(first_ids + second_ids) <= set(gist.metadata["source_ids"])

    # 新来源同样归档且可溯源到该要点
    for cid in second_ids:
        assert system.core.get(cid) is None
        archived = system.forgotten.get(cid)
        assert archived is not None
        assert archived.metadata["consolidated_into"] == gist_id
        assert gist.associations.get(cid, 0.0) > 0


def test_unrelated_cluster_still_creates_new_gist(tmp_path):
    """主题不同的新簇不被既有要点吞并（阈值高于聚类阈值）"""
    system = _make_system(tmp_path)
    _add_cluster(system, n=3)  # 老王/工作/项目
    report1 = system.sleep()
    assert len(report1.gists_created) == 1

    for i in range(3):
        system.add_memory(
            content=f"和小李第{i}次去崇礼滑雪，雪况很好",
            persons=["小李"], topics=["滑雪", "旅行"], keywords=["滑雪"],
            importance=0.6,
        )
    report2 = system.sleep()
    assert len(report2.gists_created) == 1, "不同主题被错误归并进既有要点"
    assert report2.gists_reinforced == []


def test_reinforcement_refreshes_inverted_index_and_timestamps(tmp_path):
    """强化并入的新锚点必须进核心层倒排索引（对抗审查实证缺陷）

    裸 _store.put 不更新 topic_index/person_index，常驻 MCP 进程里
    新锚点会检索不到直到重启；修复后走 core.add 的快照回滚重建。
    """
    system = _make_system(tmp_path)
    _add_cluster(system, n=3)
    report1 = system.sleep()
    gist_id = report1.gists_created[0]
    before_updated = system.core.get(gist_id).updated_at

    # 同主题新簇，但带一个全新主题"预算"
    for i in range(3):
        system.add_memory(
            content=f"和老王开会讨论了第{i}季度预算",
            persons=["老王"], topics=["工作", "项目", "预算"],
            keywords=["会议", "预算"], importance=0.6,
        )
    report2 = system.sleep()
    assert report2.gists_reinforced == [gist_id]

    gist = system.core.get(gist_id)
    assert "预算" in gist.topics
    # 新锚点立即可经倒排索引检索（不等进程重启）
    assert gist_id in system.core.topic_index.get("预算", set()), (
        "强化并入的新主题未进倒排索引"
    )
    assert gist.updated_at > before_updated, "强化未刷新 updated_at"
