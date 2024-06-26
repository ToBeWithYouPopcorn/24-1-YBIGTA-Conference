from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

# 불러올 모델과 토크나이저의 경로
model_path = "./fine_tuned_kobert_sts"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# GPU를 사용할 수 있는 경우
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Inference 함수 정의
def evaluate_similarity(sentence1, sentence2, tokenizer, model):
    # 문장을 토큰화하고 인코딩
    inputs = tokenizer(sentence1, sentence2, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
    inputs.to(device)
    
    # 추론
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    similarity_score = probabilities[:, 1].item()  # 클래스 1의 확률을 유사도 점수로 사용
    
    return similarity_score

# 두 질문 예시
question1 = "제시문 [가]를 기반으로, 어떤 사람들은 유행을 따라가는 것이 아니라 이들을 따르는 사람들이 되어 새로운 것을 채택하는 경향이 있는지 설명하시오."
question2 = "제시문 (가), (나), (다)에는 유행에 대한 다양한 관점이 포함되어 있다. 그 관점을 비교, 분석하시오. "

# 유사도 평가
predicted_score = evaluate_similarity(question1, question2, tokenizer, model)
print(f"Predicted similarity score: {predicted_score}")
