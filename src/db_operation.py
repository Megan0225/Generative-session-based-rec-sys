from neo4j import GraphDatabase
import numpy as np

uri = "bolt://localhost:7688"
username = "neo4j"
password = ""

def clear_database():
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")

def load_nodes():
    query = """
    :auto LOAD CSV with headers FROM 'file:///products_train_UK.csv' AS row
    CALL {
        WITH row
        MERGE (p:Item {item_id: row.id, 
                        title: row.title,
                        desc: row.desc        
                        locale: row.locale, 
                        price: row.price, 
                        brand: COALESCE(row.brand,'NULL') , 
                        color: COALESCE(row.color, 'NULL'), 
                        size: COALESCE(row.size, 'NULL'), 
                        model: COALESCE(row.model, 'NULL'), 
                        material: COALESCE(row.material, 'NULL') , 
                        author: COALESCE(row.author, 'NULL')}); 
        } IN TRANSACTIONS OF 1000 ROWS;
        """
    with driver.session() as session:
        session.run(query)

def load_relationships():
    query = """
    :auto LOAD CSV with headers FROM 'file:///transactions.csv' AS row
    CALL {
        WITH row
        MATCH (a:Item{item_id: row.prev_item})
        MATCH (b:Item{item_id: row.next_item})
        MERGE (a)-[:ALSO_BUY]->(b)
        } IN TRANSACTIONS OF 1000 ROWS;
    """
    with driver.session() as session:
        session.run(query)

def update_node_properties():
    query = """
    :auto LOAD CSV with headers FROM 'file:///products_train_UK.csv' AS row
    CALL {
        WITH row
        MATCH (p:Item {item_id: row.id})
        SET p.title = row.title
        SET p.desc = row.desc
        } IN TRANSACTIONS OF 1000 ROWS;
    """
    with driver.session() as session:
        session.run(query)


def del_node_properties():
    query = """
    MATCH (p:Item)
    WHERE p.desc_embedding IS NOT NULL
    WITH p LIMIT 10000
    REMOVE p.desc_embedding
    RETURN count(p)
    """
    with driver.session() as session:
        result = session.run(query)
        for count in result:
            res = count.values()
    return res[0]

def batch_update(tx, batch):
    query = """
    UNWIND $batch AS row
    MATCH (p:Item {item_id: row.id})
    SET p.td_embedding = row.embedding
    """
    tx.run(query, batch=batch)

# 建立 Brand 节点和品牌关系
def create_brand_nodes_and_relationships():
    query = """
    :auto LOAD CSV WITH HEADERS FROM 'file:///id_brand.csv' AS row
    CALL {
        WITH row 
        WHERE row.brand IS NOT NULL AND row.brand <> ''
        MERGE (b:Brand {name: row.brand})
        WITH row, b
        MATCH (i:Item {item_id: row.id})
        MERGE (i)-[:BRAND]->(b)} IN TRANSACTIONS OF 1000 ROWS;
    """
    with driver.session() as session:
        session.run(query)
        print("Brand nodes and BRAND relationships created successfully.")


if __name__ == '__main__':
    # 连接到 Neo4j 数据库
    driver = GraphDatabase.driver(uri, auth=(username, password))

    clear_database() # 清空数据库
    load_nodes() # 导入节点数据
    load_relationships() # 导入关系数据

    # update_node_properties() # 更新节点属性
    
    # # 循环删除属性（内存问题会报错，只能这样删除了）
    # count=10000
    # while count==10000:
    #     count = del_node_properties() # 删除节点属性
    #     print(f"已删除 {count} 个节点的 desc_embedding 属性")


    # 将嵌入存入 Neo4j
    # 加载 .npy 文件
    file_path= "D:/neo4j/neo4j-community-2025.01.0-windows/neo4j-community-2025.01.0/import/"
    ids = np.load(file_path+"id.npy", allow_pickle=True)
    embeddings = np.load(file_path+"td_embedding.npy", allow_pickle=True)
    # 批量导入
    batch_size = 10000
    with driver.session() as session:
        for i in range(0, len(ids), batch_size):
            batch = [{"id": str(ids[j]), "embedding": embeddings[j].tolist()} for j in range(i, min(i + batch_size, len(ids)))]
            session.write_transaction(batch_update, batch)
    print("嵌入已批量导入到 Neo4j")

    create_brand_nodes_and_relationships()
    
    # 关闭驱动程序
    driver.close()


    