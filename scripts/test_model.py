
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 指定模型
MODEL_NAME = "Qwen/Qwen3-0.6B"

print("正在加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()

print("模型加载成功！")
print("使用设备：", model.device)
print("GPU：", torch.cuda.get_device_name(0))

# 2. 准备测试问题
messages = [
    {
        "role": "user",
        "content": "小明有x升饮用水, 小红有y=x^3-2x^2+5x+2升饮用水, 问当小明有多少升水时, 小红的水比小明的水的六倍多得最多, 请说明计算过程。"
    }
]

# 3. 构造输入，关闭思考模式
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 4. 生成回答
print("\n模型正在思考...\n")

with torch.no_grad(): #不要计算grad
    outputs = model.generate(
        **inputs,
        max_new_tokens=2560,
        do_sample=False
    )

# 5. 只提取新生成的内容
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
answer = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("========== 模型回答 ==========")
print(answer)
print("==============================")

