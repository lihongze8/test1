from thought_chain import client, FamoursBook, get_thought_steps

def batch_generate_steps(prompt: str, n: int) -> list:
    steps_list = []
    for _ in range(n):
        steps_list.append(get_thought_steps(prompt))
    return steps_list