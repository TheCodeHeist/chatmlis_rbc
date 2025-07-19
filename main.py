import torch
import gpt2_ctx as GPT2
import uuid
import json
from flask import Flask, jsonify, request, send_from_directory
import sys

# -----------------------------------------------------------------------------
init_from = "gpt2-medium"
out_dir = "out"
chat_state = """You are a helpful assistant. Your name is ChatMLIS. You were made by MLIS Robotics Club of Maple Leaf International School, Dhaka, Bangladesh. You will respond to the user's questions or requests in a friendly and informative manner. Answers should be clear and concise, providing the necessary information without overwhelming the user. If the user asks for help with a specific topic, provide relevant details or examples to assist them effectively. Once you answer, invite the user ask a different question.

[Assistant]

Hello! I'm here to assist you. What would you like to know or discuss today?

[User]

Hey, what's your name?

[Assistant]

My name is ChatMLIS. I'm here to help you with any questions you may have! If you have a specific topic or question in mind, feel free to ask, and I'll do my best to assist you!

[User]

"""
seed = 1337
device = "cuda"
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)
compile = False
# exec(open('configurator.py').read())
# -----------------------------------------------------------------------------


num_samples = 5
max_new_tokens = 100
temperature = 0.8
top_k = 200

app = Flask(__name__)


print("Starting ChatMLIS...")

# Compile and run the model
model, ctx, encode, decode = GPT2.compile_and_run(
    init_from=init_from,
    out_dir=out_dir,
    seed=seed,
    device=device,
    dtype=dtype,
    compile=compile,
)
print("Model and context prepared.")
print("------------------------\n\n")


# Path for our main Svelte page
@app.route("/")
def base():
    return send_from_directory("client/build", "index.html")


# Path for all the static files (compiled JS/CSS, etc.)
@app.route("/<path:path>")
def home(path):
    return send_from_directory("client/build", path)


@app.route("/ask", methods=["POST"])
def ask():
    global chat_state, model, ctx, encode, decode

    question = request.json["question"]
    print(f"Received question: {question}")

    if not question:
        return {"error": "No question provided"}, 400

    chat_state += f"{question}\n\n[Assistant]"
    combinations = []

    chat_text_ids = encode(chat_state)
    x = torch.tensor(chat_text_ids, dtype=torch.long, device=device)[None, ...]

    # Run generation
    with torch.no_grad():
        with ctx:
            for _ in range(num_samples):
                chat_hist = chat_state

                while chat_hist.count("[User]") < chat_state.count("[User]") + 1:
                    y = model.generate(
                        x,
                        max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                    )

                    # generated = decode(y[0].tolist())
                    # chat_hist += generated[len(chat_hist) - 1 :]
                    chat_hist = decode(y[0].tolist())

                    # print("---------------------------------")
                    # print("current chat history:")
                    # print(generated[len(chat_hist) - 1 :])
                    # print("---------------------------------")

                combinations.append(
                    chat_hist[len(chat_state) + 2 :].split("\n\n")[0].strip()
                )

    response = {"combinations": combinations, "chat_state": chat_state}

    return jsonify(response)


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) > 0 and args[0] == "server":
        print("Starting ChatMLIS in web server mode...")
        print("You can access the chat interface at http://localhost:5000")

        app.run(debug=True, port=5000)
    else:
        print("Starting chat in terminal mode...")
        print("Type 'exit', 'quit', or 'stop' to end the chat.")
        print("---------------------------------")
        print(
            "You can also run the server with 'python main.py server' to use the web interface."
        )

        while True:
            question = input("[You]: ")
            if question.lower() in ["exit", "quit", "stop"]:
                print("Exiting the chat. Goodbye!")
                break

            chat_state += f"{question}\n\n[Assistant]"
            combinations = []

            chat_text_ids = encode(chat_state)
            x = torch.tensor(chat_text_ids, dtype=torch.long, device=device)[None, ...]

            # Run generation
            with torch.no_grad():
                with ctx:
                    for _ in range(num_samples):
                        chat_hist = chat_state

                        while (
                            chat_hist.count("[User]") < chat_state.count("[User]") + 1
                        ):
                            y = model.generate(
                                x,
                                max_new_tokens,
                                temperature=temperature,
                                top_k=top_k,
                            )

                            # generated = decode(y[0].tolist())
                            # chat_hist += generated[len(chat_hist) - 1 :]
                            chat_hist = decode(y[0].tolist())

                            # print("---------------------------------")
                            # print("current chat history:")
                            # print(generated[len(chat_hist) - 1 :])
                            # print("---------------------------------")

                        combinations.append(
                            chat_hist[len(chat_state) + 2 :].split("\n\n")[0].strip()
                        )

            print("")
            for i, res in enumerate(combinations):
                print(f"{i + 1}: {res}")

            response_id = -1
            while response_id < 0 or response_id >= len(combinations):
                response_id = int(input("Select a response by number (1-5): ")) - 1

                if response_id < 0 or response_id >= len(combinations):
                    print("\nInvalid selection. Please try again.")

            chat_state += f"\n\n{combinations[response_id]}\n\n[User]\n\n"

            print(f"\n[Assistant]: {combinations[response_id]}\n")

        with open(f"chat_history/{str(uuid.uuid4())}.txt", "w", encoding="utf-8") as f:
            f.write(chat_state)
            print("Chat history saved.")
            print("You can find it in the 'chat_history' directory.")

        print("Thank you for chatting with me! Have a great day!")
        print("---------------------------------")
