import requests
import pandas as pd
from tqdm import tqdm
import io

# 打开基因列表文件
with open('../data/processed_data/primekg_gene_list.txt', 'r') as f:
    primekg_gene_list = f.readlines()

primekg_gene_list = [gene.strip() for gene in primekg_gene_list]

# 假设你的基因名称列表如下
gene_list = primekg_gene_list

# 定义物种ID为人类
taxonomy_id = "9606"

# 准备存储结果的列表
results = []

def save_results(results):
    results_df = pd.DataFrame(results, columns=["Gene Name", "UniProt ID", "Sequence"])
    results_df.to_csv("../data/processed_data/uniprot_results.csv", index=False)
    print("Results saved to CSV.")

try:
    # 遍历基因名称列表并查询UniProt API
    for gene_name in tqdm(gene_list, desc="Fetching data from UniProt"):
        # 构建UniProt API URL
        uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}+AND+taxonomy_id:{taxonomy_id}&fields=accession,sequence&format=tsv"
        
        # 发送HTTP请求获取数据
        response = requests.get(uniprot_url)
        
        # 检查响应状态
        if response.status_code == 200:
            # 解析响应内容
            data = response.text
            # 检查是否返回了数据
            if data.strip():  # 如果返回了数据
                # 将TSV格式的数据转换为Pandas DataFrame
                df = pd.read_csv(io.StringIO(data), sep='\t')
                
                # 遍历DataFrame中的每一行并存储结果
                for index, row in df.iterrows():
                    sequence = row['Sequence']
                    # 检查序列长度是否足够长
                    results.append([gene_name, row['Entry'], sequence])
            else:  # 如果没有返回数据，添加一个空行
                results.append([gene_name, "", ""])
        else:
            print(f"Failed to retrieve data for gene: {gene_name}, status code: {response.status_code}")
            # 添加一个空行
            results.append([gene_name, "", ""])

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
    save_results(results)  # 保存当前获取到的数据

# 在完成所有请求后保存结果
save_results(results)