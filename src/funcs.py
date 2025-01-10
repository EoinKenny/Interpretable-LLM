import re


def pre_process_df(df, dataset):
    
    if dataset == 'imdb':
        df = df.rename(columns={'review': 'text', 'sentiment': 'label'})
    
    elif dataset == 'emails':
        df = df.rename(columns={'spam': 'label'})
        df['label'] = df['label'].map({0: 'normal', 1: 'spam'})
        
    elif dataset == 'blogs':
        df = df.rename(columns={'gender': 'label'})
    else:
        raise TypeError('invalid dataset name')
    
    return df


def make_prompt(string, dataset):

    if dataset == 'blogs':

        prompt = """
        Pretend you are an AI assistant and you are very good at doing gender classification from blog posts.
        Your task is to carefully read a blog post, and then classify the gender of the person who wrote it.
        
        You are only allowed to choose one of the following two categories in your classification: 
        1. male
        2. female
    
    
        Now, consider the following: 

        **Paragraph starting now**
        
        """+string+"""
        
        **Paragraph now ended**
                    
        First, I want you to analyze the paragraph I just showed you step by step, then, at the end of your analysis, I want you to classify it by writing either:
        
        <answer>male</answer> or <answer>female</answer> 

        Try your best, you must pick one or the other.
        """
    
    elif dataset == 'emails':

        prompt = """
        You are an AI assistant and you are very good at doing spam email classification.
        Your task is to carefully read an email, and then classify if it is spam or not.
        
        You are only allowed to choose one of the following two categories in your classification: 
        1. spam
        2. normal
    
    
        Now, consider the following: 

        **Paragraph starting now**
        
        """+string+"""
        
        **Paragraph now ended**
                    
        First, I want you to analyze the paragraph I just showed you step by step, then, at the end of your analysis, I want you to classify it by writing either:
        
        <answer>spam</answer> or <answer>normal</answer> 

        Try your best, you must pick one or the other.
        """
    
    elif dataset == 'imdb':

        prompt = """
        You are an AI assistant and you are very good at doing sentiment classification of movie reviews.
        Your task is to carefully read a movie review and then classify if the reviewer thinks negaitvely or positively about it.
        
        You are only allowed to choose one of the following two categories in your classification: 
        1. negative
        2. positive
    
        Now, consider the following: 

        **Paragraph starting now**
        
        """+string+"""
        
        **Paragraph now ended**
                    
        First, I want you to analyze the paragraph I just showed you step by step, then, at the end of your analysis, I want you to classify it by writing either:
        
        <answer>negative</answer> or <answer>positive</answer> 

        Try your best, you must pick one or the other.
        """

    else:
        raise TypeError('invalid dataset name')
    
    messages = [
    # {"role": "system", "content": "You are a useful AI chatbot who follows instructions carefully."},
    {"role": "user", "content": prompt},
    ]

    return messages



def parse_concepts(string):

    prompt = """
    You are an AI assistant and you are very good at identifying key concepts in a sequence of text.

    The text you will see is a user prompting an LLM to analyse the sentiment of a movie review.

    Your task is to extract the core concepts used in the LLM's analysis.
    
    Now, consider the following sequence of text: 
    ----------------------------------------------------
    """+string+"""
    ----------------------------------------------------

    Analyse the text step by step, and finally give the core concepts used by the LLM in the analysis of the review formatted in bullet points with
    * concept 1
    * concept 2
    * etc...
    """

    messages = [
    # {"role": "system", "content": "You are a useful AI chatbot who follows instructions carefully."},
    {"role": "user", "content": prompt},
    ]

    return messages


def extract_answer(text):
    # Pattern to match any content between <answer> tags
    pattern = r'<answer>(.*?)</answer>'
    
    # Search for the pattern in the text, allowing for multiline matching
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        # Return the matched content with whitespace stripped
        return match.group(1).strip()
    else:
        return None

