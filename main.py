from agents.code_check_agent import check_code

if __name__ == "__main__":
    print("코드 검수 시작")

    user_code = """
import torch
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# 1. 입력 문장
sentences = [
   "I love artificial intelligence.",
   "Machine learning is fascinating.",
   "Natural language processing is a subfield of AI.",
   "Deep learning improves neural networks.",
   "Transformers have changed AI forever."
]
# 2. BERT-base (순수 BERT) 준비
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
# 3. Sentence-BERT 준비
sbert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
sbert_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# 4. BERT-base 임베딩 (CLS 토큰)
def get_bert_embeddings(sentences):
   embeddings = []
   for sentence in sentences:
       inputs = bert_tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
       with torch.no_grad():
           outputs = bert_model(**inputs)
       cls_embedding = outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_size)
       embeddings.append(cls_embedding.squeeze(0).numpy())
   return np.array(embeddings)
# 5. Sentence-BERT 임베딩 (CLS 토큰)
def get_sbert_embeddings(sentences):
   inputs = sbert_tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
   with torch.no_grad():
       outputs = sbert_model(**inputs)
   cls_embeddings = outputs.last_hidden_state[:, 0, :] # (batch_size, hidden_size)
   return cls_embeddings.numpy()
# 6. 임베딩 추출
bert_embeddings = get_bert_embeddings(sentences)
sbert_embeddings = get_sbert_embeddings(sentences)
# 7. PCA로 3D 축소
pca_bert = PCA(n_components=3)
pca_sbert = PCA(n_components=3)
bert_3d = pca_bert.fit_transform(bert_embeddings)
sbert_3d = pca_sbert.fit_transform(sbert_embeddings)
# 8. 3D 시각화
fig = plt.figure(figsize=(16, 7))
# BERT 결과
ax1 = fig.add_subplot(121, projection='3d')
for i, sentence in enumerate(sentences):
   x, y, z = bert_3d[i]
   ax1.scatter(x, y, z, label=f"{i+1}")
   ax1.text(x, y, z, f'{i+1}', fontsize=9)
ax1.set_title("BERT-base (bert-base-uncased) Embeddings")
ax1.set_xlabel('PCA1')
ax1.set_ylabel('PCA2')
ax1.set_zlabel('PCA3')
# Sentence-BERT 결과
ax2 = fig.add_subplot(122, projection='3d')
for i, sentence in enumerate(sentences):
   x, y, z = sbert_3d[i]
   ax2.scatter(x, y, z, label=f"{i+1}")
   ax2.text(x, y, z, f'{i+1}', fontsize=9)
ax2.set_title("Sentence-BERT (all-MiniLM-L6-v2) Embeddings")
ax2.set_xlabel('PCA1')
ax2.set_ylabel('PCA2')
ax2.set_zlabel('PCA3')
plt.tight_layout()
plt.show()
"""

    result = check_code(user_code)
    print("🧠 코드 검수 결과:\n")
    print(result)
