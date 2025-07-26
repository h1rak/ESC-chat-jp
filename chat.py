from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def transrate(prompt):
    messages = [
        {"role": "system", "content":"あなたは優秀な翻訳者です。与えられた文章を日本語のニュアンスを残したまま、英語に翻訳してください。翻訳だけをすればいいです。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # 思考モードと非思考モードを切り替え、デフォルトは True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # テキスト補完の実行
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # 思考コンテキストのパース
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("content:", content)
    return content


strategy_history = ""
history = []
model_name = "/workspace/santa/ESC-chat/model/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

# モデルのパスとトークナイザーを指定
#model_path = '/workspace/santa/MultiESC/MultiESC/output/checkpoint-15450'  # 訓練済みモデルが保存されているディレクトリ
model_path = "/workspace/santa/ESC-chat/checkpoint-14523"
strategy_tokenizer = BartTokenizer.from_pretrained(model_path)
strategy_model = BartForConditionalGeneration.from_pretrained(model_path)

while True:
    # モデルへの入力
    prompt = input("入力：")
    history.append(prompt)
    content = transrate(prompt)

    strategy_history += content
    # モデルを推論モードに設定
    strategy_model.eval()
    inputs = strategy_tokenizer(strategy_history, padding=True, truncation=True, return_tensors="pt")
    #print("thinking content:", thinking_content)

    inputs = strategy_tokenizer(content, return_tensors="pt")
    output = strategy_model.generate(input_ids=inputs["input_ids"], max_length=512, num_beams=4)
            
    # デコード
    decoded_output = strategy_tokenizer.decode(output[0], skip_special_tokens=True)
    # 進行度を表示
    print(f"Input: {strategy_history}")
    print(f"Generated Output: {decoded_output}")
    print("###")
    strategy_history += strategy_tokenizer.pad_token + decoded_output

    def create_strategy_messages(history,strategy, user_prompt):
        """
        会話戦略に基づいてメッセージを生成する関数
        
        Args:
            strategy (str): 選択された会話戦略
            user_prompt (str): ユーザーのプロンプト
        
        Returns:
            list: メッセージのリスト
        """
        
        # 会話戦略の定義
        strategies = {
            "@[Question]": "質問を通じて相手の考えや気持ちを深く理解し、対話を促進する戦略",
            "@[Greeting]": "親しみやすい挨拶や雰囲気作りを通じて、良好な関係性を築く戦略", 
            "@[Restatement or Paraphrasing]": "相手の発言を言い換えて確認し、理解を示す戦略",
            "@[Reflection of feelings]": "相手の感情を察知し、共感的に反映する戦略",
            "@[Self-disclosure]": "適切な自己開示を通じて親近感を生み出す戦略",
            "@[Affirmation and Reassurance]": "相手を肯定し、安心感を与える戦略",
            "@[Providing Suggestions or Information]": "有用な提案や情報を提供する戦略",
            "@[Others]": "その他の柔軟なアプローチを用いる戦略"
        }
        history = " ".join(history)
        
        # システムメッセージを生成
        system_content = f"""以下の会話戦略に従って返答を生成してください：

        選択された戦略: {strategy}
        戦略の説明: {strategies.get(strategy, "未定義の戦略")}

        この戦略に基づいて、ユーザーのメッセージに対して適切で効果的な返答を生成してください。
        返答は自然で人間らしく、相手との良好なコミュニケーションを促進するものにしてください。"""

        # メッセージ構造を作成
        messages = [
            {"role":"system","content":history},
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages

    messages = create_strategy_messages(history,decoded_output,prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # 思考モードと非思考モードを切り替え、デフォルトは True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # テキスト補完の実行
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # 思考コンテキストのパース
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("content:", content)
    history.append(decoded_output + content)
    content = transrate(content)

    strategy_history += strategy_tokenizer.pad_token + content
    print("strategy_history:",strategy_history)
    print("history:",history)