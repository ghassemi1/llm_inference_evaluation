# PROMPTS = [
#     "Explain transformers",
#     "Write JSON with 3 keys",
#     "Summarize AI in 2 lines",
# ] * 20

# print(PROMPTS)
requests = [[1, 5, 5, 0], [2, 7, 8, 1], [3, 7, 5, 1], [4, 10, 3, 3]]
req = sorted(requests, key=lambda x: (x[2], -x[3]), reverse=True)
print(req)
