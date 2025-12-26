import pandas as pd

# 读取CSV
df = pd.read_csv('example_data/judge_contract_review.csv')

print("CSV列名：")
print(list(df.columns))
print("\nCSV行数：", len(df))
print("\n前3行数据：")
print(df.head(3))

# 检查是否有隐藏字符
for col in df.columns:
    print(f"\n列名: '{col}' (长度: {len(col)}, repr: {repr(col)})")
