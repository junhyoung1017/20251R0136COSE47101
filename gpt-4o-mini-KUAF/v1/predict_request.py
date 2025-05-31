from openai import OpenAI
import os

import json


def read_jsonl_to_messages(file_path, max_count=1000):
    """
    JSONL 파일을 읽어서 문자열 message 배열로 변환

    Args:
        file_path (str): JSONL 파일 경로
        max_count (int): 읽을 최대 줄 수 (기본값: 1000)

    Returns:
        list: message 문자열들의 배열
    """
    messages = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if i >= max_count:
                    break

                # 빈 줄 건너뛰기
                if not line.strip():
                    continue

                try:
                    # JSON 파싱
                    data = json.loads(line.strip())

                    # 'message' 필드가 있는 경우 추출
                    if 'message' in data:
                        messages.append(str(data['message']))
                    # 'message' 필드가 없으면 전체 객체를 문자열로 변환
                    else:
                        messages.append(json.dumps(data, ensure_ascii=False))

                except json.JSONDecodeError as e:
                    print(f"줄 {i + 1}에서 JSON 파싱 오류: {e}")
                    continue

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
        return []

    return messages


# 사용 예제
if __name__ == "__main__":
    # JSONL 파일 경로 지정
    file_path = "github_profiles_v5_test_without_stack.jsonl"  # 실제 파일 경로로 변경하세요

    # 메시지 배열 생성 (최대 1000개)
    message_array = read_jsonl_to_messages(file_path, max_count=300)

    # 결과 출력
    print(f"총 {len(message_array)}개의 메시지를 읽었습니다.")

    # 처음과 끝 보기
    #print(message_array[0][0:30])
    #print(message_array[209]+"이 user의 stack을 예측해줘")


    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    comp = []

    for i in range(len(message_array)):
        temp = client.chat.completions.create(
            model="ft:gpt-4o-mini-2024-07-18:personal:kuaf:BdCKIwgY",
            messages=[
                {"role": "user", "content": message_array[i]+" 이 user의 stack을 예측해줘."}
            ]
        )
        comp.append(temp)
        print(message_array[i][0:30])
        print(comp[i].choices[0].message.content)
        print()
        print()


