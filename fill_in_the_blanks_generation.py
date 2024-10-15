
async def generate_fill_in_the_blank_questions(context,answer):
    answerSize = len(answer)
    replacedBlanks = ""
    for i in range(answerSize):
        replacedBlanks += "_"
    blank_q = context.replace(answer,replacedBlanks)
    return blank_q