from neo4j import GraphDatabase
from typing import List, Literal, Dict

uri = "bolt://localhost:7688"
username = "neo4j"
password = "Lm15420202202206"

class GraphReRanker:
    def __init__(self, driver):
        self.driver = driver

    def rerank(
        self,
        user_history: List[str],
        candidate_items: List[str],
        method: Literal["appr", "path_score"] = "path_score",
        top_k: int = 20,
        alpha: float = 0.85,
        rule_weights: Dict[str, float] = None
    ) -> List[str]:
        if method == "appr":
            return self._rerank_with_appr(user_history, candidate_items, top_k, alpha)
        elif method == "path_score":
            return self._rerank_with_path_scoring(user_history, candidate_items, rule_weights, top_k)
        else:
            raise ValueError("Unsupported rerank method")

    def _rerank_with_appr(self, user_history, candidate_items, top_k, alpha):
        query = f"""
        CALL gds.pageRank.stream('itemGraph', {{
            maxIterations: 20,
            dampingFactor: {alpha},
            sourceNodes: [n IN $history | gds.util.asNode(n)]
        }})
        YIELD nodeId, score
        WITH gds.util.asNode(nodeId) AS item, score
        WHERE item.item_id IN $candidates
        RETURN item.item_id AS item_id, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self.driver.session() as session:
            result = session.run(query, history=user_history, candidates=candidate_items, top_k=top_k)
            return [record["item_id"] for record in result]

    def _rerank_with_path_scoring(self, user_history, candidate_items, rule_weights, top_k):
        # 默认规则及其权重
        if rule_weights is None:
            rule_weights = {
                "also_buy": 1.0,
                "same_brand": 1.0
            }

        rules = {
            "also_buy": """
                MATCH (h:Item)-[:ALSO_BUY]->(c:Item)
                WHERE h.item_id IN $history AND c.item_id IN $candidates
                RETURN c.item_id AS item_id, count(*) * $w AS score
            """,
            "same_brand": """
                MATCH (h:Item)-[:BRAND]->(b:Brand)<-[:BRAND]-(c:Item)
                WHERE h.item_id IN $history AND c.item_id IN $candidates
                RETURN c.item_id AS item_id, count(*) * $w AS score
            """
        }

        scores = {}
        with self.driver.session() as session:
            for rule_name, cypher in rules.items():
                result = session.run(cypher, history=user_history, candidates=candidate_items, w=rule_weights[rule_name])
                for record in result:
                    item = record["item_id"]
                    score = record["score"]
                    scores[item] = scores.get(item, 0) + score

        # 按照得分排序
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:top_k]]

if __name__ == "__main__":
    # 连接到 Neo4j 数据库
    driver = GraphDatabase.driver(uri, auth=(username, password))

    # 初始化
    reranker = GraphReRanker(driver)

    # 输入用户历史和候选商品
    user_history = ['B007MO0FIO', 'B09QXPN2TL', 'B005PA3I8G', 'B003Y7MS6U', 'B005PA3I8G', 'B003Y7MS6U', 'B007MO0FIO'],
    candidate_items = ['B07YSZZYBB', 'B075R6ZKZZ', 'B09J3NT7G5', 'B082ZFY7T7', 'B00DIBEFZY', 'B00CYA5QXK', 'B00CXAOTCK', 'B07W68J6BV', 'B09KN93SYN', 'B07W46JLMX', 'B0081YFKY8', 'B078WXY5NM', 'B00FF78WEE', 'B08YR27JJ8', 'B017V1CTH2', 'B00QH12GSU', 'B07HF8X4M1', 'B09X6VV6VY', 'B00KYQ9UC4', 'B08F7VD6CD']

    # # 使用 APPR 重排序
    # top_items_appr = reranker.rerank(user_history, candidate_items, method="appr", top_k=5)
    # print(top_items_appr)

    # 使用路径规则重排序
    top_items_path = reranker.rerank(user_history, candidate_items, method="path_score", top_k=5)

    # 关闭驱动程序
    driver.close()

