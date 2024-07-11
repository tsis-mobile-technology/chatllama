import os
import google.generativeai as genai
from typing import List, Tuple
import argparse

from chatllama.langchain_modules.prompt_templates import (
    PERSON_CHATBOT_TEMPLATE,
    AI_CHATBOT_TEMPLATE,
)
# # 프롬프트 템플릿 정의
# PERSON_CHATBOT_TEMPLATE = """
# 당신은 인간입니다. AI 챗봇과 대화를 하고 있습니다. 
# 이전 대화: {history}
# AI 챗봇: {chatbot_input}
# 인간으로서 자연스럽게 응답하세요:
# """

# AI_CHATBOT_TEMPLATE = """
# 당신은 AI 챗봇입니다. 인간과 대화를 하고 있습니다.
# 이전 대화: {history}
# 인간: {human_input}
# AI 챗봇으로서 친절하고 도움이 되는 응답을 제공하세요:
# """

# Gemini API 키 설정
os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY_HERE'  # 실제 API 키로 교체해야 함
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


CONVERSATION_LENGTH = 20

def create_gemini_model():
    return genai.GenerativeModel('gemini-pro')

def generate_response(model: genai.GenerativeModel, prompt: str) -> str:
    response = model.generate_content(prompt)
    return response.text

def create_conversation(model: genai.GenerativeModel) -> List[str]:
    conversation = []
    history = ""
    for i in range(CONVERSATION_LENGTH):
        # 인간 응답 생성
        human_prompt = PERSON_CHATBOT_TEMPLATE.format(
            history=history,
            chatbot_input="" if i == 0 else conversation[-1].split(": ", 1)[1]
        )
        human_output = generate_response(model, human_prompt)
        conversation.append(f"Human: {human_output}")

        # AI 챗봇 응답 생성
        bot_prompt = AI_CHATBOT_TEMPLATE.format(
            history=history,
            human_input=human_output
        )
        bot_output = generate_response(model, bot_prompt)
        conversation.append(f"AI: {bot_output}")

        # 히스토리 업데이트 (최근 4개 턴만 유지)
        history = "\n".join(conversation[-8:])

    return conversation

def main(num_conversations: int, output_file: str):
    model = create_gemini_model()
    all_conversations = []

    for i in range(num_conversations):
        print(f"생성 중인 대화: {i+1}/{num_conversations}")
        conversation = create_conversation(model)
        all_conversations.append("\n".join(conversation))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n\nNEW CONVERSATION\n\n".join(all_conversations))

    print(f"{num_conversations}개의 대화가 {output_file}에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini를 사용한 대화 생성기")
    parser.add_argument("--num_conversations", type=int, default=10, help="생성할 대화의 수")
    parser.add_argument("--output_file", type=str, default="conversations.txt", help="출력 파일 이름")
    args = parser.parse_args()

    try:
        main(args.num_conversations, args.output_file)
    except Exception as e:
        print(f"오류 발생: {e}")