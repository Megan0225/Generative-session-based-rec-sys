import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI
import ast
from rerank import GraphReRanker

# ---------- 配置区域 ----------
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = ""

AZURE_OPENAI_API_KEY = ""
AZURE_OPENAI_ENDPOINT = "https://hkust.azure-api.net"
AZURE_OPENAI_VERSION = "2023-05-15"

# ---------- 初始化模型和连接 ----------
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
client = AzureOpenAI(api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# ---------- 核心模块 ----------

def gpt3_5(msg):
    response = client.chat.completions.create(
        model="gpt-35-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg}
        ],
        max_tokens=200
    )
    return response.choices[0].message.content

def get_item_attributes(item_ids):
    query = """
    MATCH (i:Item)
    WHERE i.item_id IN $item_ids
    RETURN i.title AS title, i.description AS description, i.price AS price, 
           i.brand AS brand, i.color AS color, i.size AS size, i.model AS model, i.material AS material
    """
    with driver.session() as session:
        result = session.run(query, item_ids=item_ids)
        return [record.data() for record in result]

def generate_user_intent_description(prev_item_attributes):
    prompt_template =  """
            Based on the following attributes, describe the user’s current intent in natural language, so we can retrieve items that match their next interest.
            Titles and descriptions of previous purchases: {t_d}
            Preferred colors: {colors}
            Price range: {price_range}
            Brands: {brands}
            Models: {models}
            Materials: {materials}
            """
    # prompt_template = """
    #         Given the following purchase history and product attributes, infer what the user might be trying to accomplish or solve. Output only one sentence that describes the user's future need, to be used for retrieving semantically relevant products.
    #         Purchase details:
    #         {t_d}
    #         Colors: {colors}; Price: {price_range}; Brands: {brands}; Models: {models}; Materials: {materials}
    #         """
    msg = prompt_template.format(
        t_d="; ".join(f"{a['title']} ({a.get('description', '')})" for a in prev_item_attributes),
        colors="; ".join(a['color'] for a in prev_item_attributes if a.get('color')),
        price_range="{} - {}".format(
            min(a['price'] for a in prev_item_attributes if a.get('price') is not None),
            max(a['price'] for a in prev_item_attributes if a.get('price') is not None)
        ) if any(a.get('price') for a in prev_item_attributes) else "unknown",
        brands="; ".join(a['brand'] for a in prev_item_attributes if a.get('brand')),
        models="; ".join(a['model'] for a in prev_item_attributes if a.get('model')),
        materials="; ".join(a['material'] for a in prev_item_attributes if a.get('material')),
    )
    return gpt3_5(msg)

def embed(text, model):
    embedding = model.encode([text])[0]
    return embedding.tolist()

def vector_query_topk(driver, embedding, k=5):
    query = """
    CALL db.index.vector.queryNodes('title_description', $k, $embedding)
    YIELD node, score
    RETURN node.item_id AS item_id, node.title AS title, score
    ORDER BY score DESC
    """
    with driver.session() as session:
        result = session.run(query, {"k": k, "embedding": embedding})
        return [record.data() for record in result]

def recommend_from_session(prev_items, k=10):
    # print(f"\n🧩 Testing session with items: {prev_items}")
    attributes = get_item_attributes(prev_items)
    if not attributes:
        print("❌ Failed to retrieve item attributes.")
        return
    user_intent_text = generate_user_intent_description(attributes)
    # print("\n🧠 User Intent Description:\n", user_intent_text)

    intent_embedding = embed(user_intent_text, model)
    topk_results = vector_query_topk(driver, intent_embedding, k)
    # topk_ids = [item["item_id"] for item in topk_results]

    # print(f"\n✅ Top-{k} Recommended Items:")
    # for item in topk_results:
    #     print(f"- {item['title']} (score: {item['score']:.4f}) [ID: {item['item_id']}]")

    return topk_results, user_intent_text

def compute_mrr(results, targets):
    ranks = []
    for recs, target in zip(results, targets):
        try:
            rank = recs.index(target) + 1
            ranks.append(1.0 / rank)
        except ValueError:
            ranks.append(0.0)
    return sum(ranks) / len(ranks)

def evaluate(top_k=20, result_path="result/result.csv", rerank=False):
    df = pd.read_csv("data\sessions_test_task1_phase1_UK_100.csv")
    gt = pd.read_csv("data\gt_task1_UK_100.csv")

    all_preds = []
    all_truth = gt.next_item.tolist()
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            prev_items = ast.literal_eval(row["prev_items"].replace("\n", "").replace(" ", ","))
            topk_results, user_intent_text = recommend_from_session(prev_items, k=top_k)
            top_k_items = [item["item_id"] for item in topk_results]
            all_preds.append(top_k_items)

            rows.append({
                "top_k_predictions": top_k_items,
                "user_intent": user_intent_text,
            })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue

    # 保存结果
    pd.DataFrame(rows).to_csv(result_path, index=False)

    # MRR
    mrr = compute_mrr(all_preds, all_truth)
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")

    # 召回率
    recall = sum([1 for pred, truth in zip(all_preds, all_truth) if truth in pred]) / len(all_truth)
    print(f"Recall@{top_k}: {recall:.4f}")

def evaluate_rerank(top_k=20,file_path="result/result_100_k50.csv", result_path="result/result_100_k50_rerank.csv"):
    df = pd.read_csv(file_path)
    session_df = pd.read_csv("data/sessions_test_task1_phase1_UK_100.csv")
    gt = pd.read_csv("data/gt_task1_UK_100.csv")

    reranker = GraphReRanker(driver)

    all_preds = []
    all_truth = gt.next_item.tolist()
    rows = []

    for row1, row2 in tqdm(zip(df.iterrows(),session_df.iterrows()), total=len(df)):
        try:
            top_k_items = ast.literal_eval(row1[1]["top_k_predictions"])
            user_history = ast.literal_eval(row2[1]["prev_items"].replace("\n", "").replace(" ", ","))
            top_k_items = reranker.rerank(user_history, top_k_items, method="path_score", top_k=top_k)

            all_preds.append(top_k_items)

            rows.append({
                "top_k_predictions": top_k_items,
                "prev_items": user_history
            })
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    # 保存结果
    pd.DataFrame(rows).to_csv(result_path, index=False)

    # MRR
    mrr = compute_mrr(all_preds, all_truth)
    print(f"\nMean Reciprocal Rank (MRR): {mrr:.4f}")

    # 召回率
    recall = sum([1 for pred, truth in zip(all_preds, all_truth) if truth in pred]) / len(all_truth)
    print(f"Recall@{top_k}: {recall:.4f}")

# ---------- 示例运行 ----------

if __name__ == "__main__":


    # test_session = {
    #     # "prev_items": ['B0BFDL54Y7', 'B0BFDR9X13', 'B07J4WF8VH', 'B07Y21LDJX'],
    #     # "next_item": 'B07Y227WNJ'
    #     "prev_items": ['B09X37JZ54', 'B0B9GH9597', 'B0B9GHMSXS', 'B074PQTR4W'],
    #     "next_item": 'B01K1SETD4'
    # }
    # try:
    #     topk_results, _ = recommend_from_session(test_session["prev_items"], k=20)
    #     topk_ids = [item["item_id"] for item in topk_results]
    #     print(test_session["next_item"] in topk_ids)
    # finally:
    #     driver.close()

    evaluate(top_k=50,result_path="result/result.csv")
    evaluate_rerank()

    # 关闭驱动程序
    driver.close()