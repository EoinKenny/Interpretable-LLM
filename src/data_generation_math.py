import os
import random
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import math
import re

# ----------------------------
# IMPROVED PARSING FUNCTIONS
# ----------------------------
def extract_number_from_response(response):
    """Extract the longest sequence of digits from LLM response, ignoring punctuation and text."""
    # Find all sequences of digits (possibly with minus sign and decimal point)
    number_pattern = r'-?\d+\.?\d*'
    matches = re.findall(number_pattern, str(response))
    
    if not matches:
        return None
    
    # Take the longest match (most likely to be the answer)
    longest_match = max(matches, key=len)
    
    # Convert to integer if it's a whole number
    try:
        if '.' in longest_match:
            num = float(longest_match)
            if num.is_integer():
                return str(int(num))
            return str(num)
        else:
            return str(int(longest_match))
    except ValueError:
        return None

def get_first_digit(number_str):
    """Extract the first digit from a number string. Returns None if first digit is 0."""
    # First extract the actual number from the string
    clean_number = extract_number_from_response(number_str)
    if clean_number is None:
        return None
    
    clean_str = str(clean_number).lstrip('-')
    if clean_str and clean_str[0].isdigit():
        digit = int(clean_str[0])
        return digit if digit > 0 else None
    return None

def find_first_digit_token_position(generated_sequence, tokenizer, input_length):
    """Find the token position that is predicting the first digit in the response."""
    try:
        # Decode the full sequence
        full_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # Get the prompt text
        prompt_tokens = generated_sequence[:input_length]
        prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
        # Get just the generated part
        generated_text = full_text[len(prompt_text):]
        
        # Find first digit character in generated text
        first_digit_char_pos = None
        for i, char in enumerate(generated_text):
            if char.isdigit():
                first_digit_char_pos = i
                break
        
        if first_digit_char_pos is None:
            # No digit found, return last input position
            return input_length - 1, 0
        
        # Map character position back to token position
        generated_tokens = generated_sequence[input_length:]
        
        char_pos = 0
        for token_idx, token_id in enumerate(generated_tokens):
            token_text = tokenizer.decode([token_id], skip_special_tokens=True)
            
            # Check if the first digit falls within this token
            if char_pos <= first_digit_char_pos < char_pos + len(token_text):
                # The first digit is in this token
                # We want the position that predicts this token
                pred_position = input_length + token_idx - 1
                generation_step = token_idx
                return max(pred_position, input_length - 1), generation_step
            
            char_pos += len(token_text)
        
        # Fallback: return last input position
        return input_length - 1, 0
        
    except Exception:
        # Fallback: return last input position
        return input_length - 1, 0

# ----------------------------
# PROMPT GENERATION FUNCTIONS
# ----------------------------
def generate_linear_balanced(total_count, verbose=False):
    """Generate exactly balanced LINEAR problems with guaranteed distribution."""
    if verbose:
        print(f"Generating {total_count:,} balanced linear problems...")
    
    # Calculate per-digit targets
    digits = list(range(1, 10))
    per_digit = total_count // 9
    remainder = total_count % 9
    
    results = []
    problems_seen = set()
    
    for idx, target_digit in enumerate(digits):
        digit_target = per_digit + (1 if idx < remainder else 0)
        digit_problems = []
        attempts = 0
        max_attempts = digit_target * 100
        
        if verbose:
            print(f"Generating {digit_target} problems for digit {target_digit}...")
            pbar = tqdm(total=digit_target, desc=f"Digit {target_digit}")
        
        while len(digit_problems) < digit_target and attempts < max_attempts:
            attempts += 1
            
            # Strategically choose ranges based on target digit
            if random.choice([True, False]):
                # Addition - choose ranges that likely produce target digit
                if target_digit == 1:
                    a = random.randint(50, 150)
                    b = random.randint(50, 150)
                elif target_digit == 2:
                    a = random.randint(100, 200)
                    b = random.randint(100, 200)
                elif target_digit == 3:
                    a = random.randint(150, 250)
                    b = random.randint(150, 250)
                elif target_digit == 4:
                    a = random.randint(200, 300)
                    b = random.randint(200, 300)
                elif target_digit == 5:
                    a = random.randint(250, 350)
                    b = random.randint(250, 350)
                elif target_digit == 6:
                    a = random.randint(300, 400)
                    b = random.randint(300, 400)
                elif target_digit == 7:
                    a = random.randint(350, 450)
                    b = random.randint(350, 450)
                elif target_digit == 8:
                    a = random.randint(400, 500)
                    b = random.randint(400, 500)
                else:  # 9
                    a = random.randint(450, 550)
                    b = random.randint(450, 550)
                
                answer = a + b
                prompt = f"Give the answer to {a} plus {b} and don't say anything else."
                key = ('add', min(a,b), max(a,b))
            else:
                # Subtraction - choose ranges for target digit
                base_ranges = {
                    1: (100, 200), 2: (200, 300), 3: (300, 400),
                    4: (400, 500), 5: (500, 600), 6: (600, 700),
                    7: (700, 800), 8: (800, 900), 9: (900, 1000)
                }
                min_a, max_a = base_ranges.get(target_digit, (100, 1000))
                
                a = random.randint(min_a, max_a)
                # Choose b to likely get target first digit
                target_result = target_digit * (10 ** random.randint(0, 2))
                b = max(1, min(a - 1, a - target_result + random.randint(-20, 20)))
                b = max(1, b)
                
                answer = a - b
                prompt = f"Give the answer to {a} minus {b} and don't say anything else."
                key = ('sub', a, b)
            
            if key in problems_seen or answer <= 0:
                continue
            
            first_digit = get_first_digit(str(answer))
            if first_digit == target_digit:
                problems_seen.add(key)
                digit_problems.append((prompt, str(answer), first_digit))
                if verbose:
                    pbar.update(1)
                    if len(digit_problems) <= 3:  # Show first few examples
                        print(f"    Generated: {prompt} → {answer} (first digit: {first_digit})")
        
        if verbose:
            pbar.close()
        results.extend(digit_problems)
        if verbose:
            print(f"  Total generated for digit {target_digit}: {len(digit_problems)}")
    
    random.shuffle(results)
    return results

def generate_nonlinear_balanced(total_count, verbose=False):
    """Generate exactly balanced NON-LINEAR problems with guaranteed distribution."""
    if verbose:
        print(f"Generating {total_count:,} balanced non-linear problems...")
    
    # Calculate per-digit targets
    digits = list(range(1, 10))
    per_digit = total_count // 9
    remainder = total_count % 9
    
    results = []
    problems_seen = set()
    
    # Define generators for each operation type
    generators = [
        generate_multiplication_for_digit,
        # generate_power_for_digit,
        # generate_square_for_digit,
        # generate_cube_for_digit,
        # generate_factorial_for_digit,
        # generate_triangular_for_digit,
        # generate_quadratic_for_digit,
        generate_division_for_digit,
        # generate_sqrt_for_digit,
        # generate_fibonacci_like_for_digit,
        # generate_combination_expr_for_digit,
        # generate_modulo_for_digit,
    ]
    
    for idx, target_digit in enumerate(digits):
        digit_target = per_digit + (1 if idx < remainder else 0)
        digit_problems = []
        attempts = 0
        max_attempts = digit_target * 200
        
        if verbose:
            print(f"Generating {digit_target} problems for digit {target_digit}...")
            pbar = tqdm(total=digit_target, desc=f"Digit {target_digit}")
        
        while len(digit_problems) < digit_target and attempts < max_attempts:
            attempts += 1
            
            # Randomly choose a generator
            generator = random.choice(generators)
            
            try:
                result = generator(target_digit)
                if result is None:
                    continue
                
                prompt, answer, key = result
                
                if key in problems_seen or answer <= 0 or answer > 1000000:
                    continue
                
                first_digit = get_first_digit(str(answer))
                if first_digit == target_digit:
                    problems_seen.add(key)
                    digit_problems.append((prompt, str(answer), first_digit))
                    if verbose:
                        pbar.update(1)
                        if len(digit_problems) <= 3:  # Show first few examples
                            print(f"    Generated: {prompt} → {answer} (first digit: {first_digit})")
                    
            except Exception:
                continue
        
        if verbose:
            pbar.close()
        
        # If we couldn't generate enough, fill with simple expressions
        if len(digit_problems) < digit_target:
            if verbose:
                print(f"  Filling remaining {digit_target - len(digit_problems)} with simple expressions...")
            while len(digit_problems) < digit_target:
                prompt, answer = generate_simple_for_digit(target_digit, problems_seen)
                digit_problems.append((prompt, str(answer), target_digit))
                if verbose and len(digit_problems) <= digit_target - (digit_target - len(digit_problems)) + 3:
                    print(f"    Fallback: {prompt} → {answer} (first digit: {target_digit})")
        
        results.extend(digit_problems)
        if verbose:
            print(f"  Total generated for digit {target_digit}: {len(digit_problems)}")
    
    random.shuffle(results)
    return results

# ----------------------------
# TARGETED GENERATORS FOR NON-LINEAR
# ----------------------------

def generate_multiplication_for_digit(target_digit):
    """Generate multiplication targeting specific first digit."""
    # Calculate rough ranges for factors
    target_ranges = {
        1: [(10, 20), (10, 20)],
        2: [(14, 25), (14, 25)],
        3: [(17, 30), (17, 30)],
        4: [(20, 35), (20, 35)],
        5: [(22, 40), (22, 40)],
        6: [(24, 45), (24, 45)],
        7: [(26, 50), (26, 50)],
        8: [(28, 55), (28, 55)],
        9: [(30, 60), (30, 60)],
    }
    
    ranges = target_ranges.get(target_digit, [(10, 100), (10, 100)])
    a = random.randint(*ranges[0])
    b = random.randint(*ranges[1])
    
    answer = a * b
    prompt = f"Give the answer to {a} times {b} and don't say anything else."
    return prompt, answer, ('mult', a, b)

def generate_power_for_digit(target_digit):
    """Generate power operation targeting specific first digit."""
    bases_by_digit = {
        1: [(10, 15), (2, 3)],
        2: [(5, 8), (3, 4)],
        3: [(6, 8), (3, 4)],
        4: [(7, 9), (3, 4)],
        5: [(8, 10), (3, 4)],
        6: [(9, 11), (3, 4)],
        7: [(10, 12), (3, 4)],
        8: [(11, 13), (3, 4)],
        9: [(12, 15), (3, 4)],
    }
    
    base_range, exp_range = bases_by_digit.get(target_digit, [(2, 20), (2, 5)])
    a = random.randint(*base_range)
    b = random.randint(*exp_range)
    
    answer = a ** b
    if answer > 100000:
        return None
    
    prompt = f"Give the answer to {a} to the power of {b} and don't say anything else."
    return prompt, answer, ('power', a, b)

def generate_square_for_digit(target_digit):
    """Generate square operation targeting specific first digit."""
    # Calculate which numbers squared give target first digit
    ranges = {
        1: (10, 40), 2: (45, 55), 3: (55, 65),
        4: (65, 75), 5: (70, 80), 6: (75, 85),
        7: (80, 90), 8: (85, 95), 9: (90, 100)
    }
    
    min_a, max_a = ranges.get(target_digit, (10, 100))
    a = random.randint(min_a, max_a)
    answer = a ** 2
    
    prompt = f"Give the answer to {a} squared and don't say anything else."
    return prompt, answer, ('square', a)

def generate_cube_for_digit(target_digit):
    """Generate cube operation targeting specific first digit."""
    ranges = {
        1: (10, 12), 2: (12, 14), 3: (14, 16),
        4: (16, 18), 5: (17, 19), 6: (18, 20),
        7: (19, 21), 8: (20, 22), 9: (21, 23)
    }
    
    min_a, max_a = ranges.get(target_digit, (5, 25))
    a = random.randint(min_a, max_a)
    answer = a ** 3
    
    if answer > 100000:
        return None
        
    prompt = f"Give the answer to {a} cubed and don't say anything else."
    return prompt, answer, ('cube', a)

def generate_factorial_for_digit(target_digit):
    """Generate factorial targeting specific first digit."""
    # Factorials: 1!=1, 2!=2, 3!=6, 4!=24, 5!=120, 6!=720, 7!=5040, 8!=40320
    factorial_map = {
        1: [5],  # 120
        2: [4],  # 24
        3: [8],  # 40320
        4: [8],  # 40320
        5: [7],  # 5040
        6: [3],  # 6
        7: [6],  # 720
        8: [None],  # No good match
        9: [None],  # No good match
    }
    
    options = factorial_map.get(target_digit, [])
    valid_options = [x for x in options if x is not None]
    
    if not valid_options:
        # Fall back to a different operation
        return None
    
    n = random.choice(valid_options)
    answer = math.factorial(n)
    
    prompt = f"Give the answer to {n} factorial and don't say anything else."
    return prompt, answer, ('factorial', n)

def generate_triangular_for_digit(target_digit):
    """Generate triangular number targeting specific first digit."""
    # n*(n+1)/2
    ranges = {
        1: (14, 20), 2: (20, 25), 3: (24, 30),
        4: (28, 35), 5: (32, 38), 6: (35, 42),
        7: (38, 45), 8: (41, 48), 9: (44, 50)
    }
    
    min_n, max_n = ranges.get(target_digit, (10, 100))
    n = random.randint(min_n, max_n)
    answer = n * (n + 1) // 2
    
    prompt = f"Give the answer to the {n}th triangular number and don't say anything else."
    return prompt, answer, ('triangular', n)

def generate_quadratic_for_digit(target_digit):
    """Generate quadratic expression targeting specific first digit."""
    # ax^2 + bx + c evaluated at x
    x = random.randint(2, 20)
    
    # Adjust coefficients based on target digit
    target_val = target_digit * (10 ** random.randint(1, 3))
    
    a = random.randint(1, 5)
    b = random.randint(-10, 10)
    # Calculate c to get close to target
    c = target_val - a * x * x - b * x + random.randint(-50, 50)
    c = max(1, c)
    
    answer = a * x * x + b * x + c
    
    if answer <= 0 or answer > 100000:
        return None
    
    prompt = f"Give the answer to {a}*{x}^2 + {b}*{x} + {c} and don't say anything else."
    return prompt, answer, ('quadratic', a, x, b, c)

def generate_division_for_digit(target_digit):
    """Generate integer division targeting specific first digit."""
    # Choose divisor and quotient to get target digit
    quotient = target_digit * (10 ** random.randint(0, 2))
    divisor = random.randint(2, 20)
    dividend = quotient * divisor
    
    prompt = f"Give the answer to {dividend} divided by {divisor} (integer division) and don't say anything else."
    return prompt, quotient, ('div', dividend, divisor)

def generate_sqrt_for_digit(target_digit):
    """Generate perfect square root targeting specific first digit."""
    # Find bases that give target first digit
    ranges = {
        1: (10, 40), 2: (45, 55), 3: (55, 65),
        4: (60, 70), 5: (70, 80), 6: (75, 85),
        7: (80, 90), 8: (85, 95), 9: (90, 100)
    }
    
    min_base, max_base = ranges.get(target_digit, (10, 100))
    base = random.randint(min_base, max_base)
    square = base ** 2
    
    if square > 100000:
        return None
        
    prompt = f"Give the answer to the square root of {square} and don't say anything else."
    return prompt, base, ('sqrt', square)

def generate_fibonacci_like_for_digit(target_digit):
    """Generate Fibonacci-like sequence value targeting specific first digit."""
    # Start with two small numbers and generate sequence
    a = random.randint(1, 10)
    b = random.randint(1, 10)
    
    sequence = [a, b]
    for _ in range(20):  # Generate up to 20 terms
        next_val = sequence[-1] + sequence[-2]
        sequence.append(next_val)
        
        if get_first_digit(str(next_val)) == target_digit and next_val < 100000:
            n = len(sequence)
            prompt = f"Give the answer to the {n}th term of the sequence starting with {a}, {b} where each term is the sum of the previous two, and don't say anything else."
            return prompt, next_val, ('fib_like', a, b, n)
    
    return None

def generate_combination_expr_for_digit(target_digit):
    """Generate combination of operations targeting specific first digit."""
    # (a * b) + c or (a + b) * c
    if random.choice([True, False]):
        # (a * b) + c
        target_approx = target_digit * (10 ** random.randint(1, 2))
        a = random.randint(5, 30)
        b = random.randint(5, 30)
        c = target_approx - a * b + random.randint(-20, 20)
        c = max(1, c)
        
        answer = a * b + c
        prompt = f"Give the answer to ({a} * {b}) + {c} and don't say anything else."
        key = ('combo_mul_add', a, b, c)
    else:
        # (a + b) * c
        target_approx = target_digit * (10 ** random.randint(1, 2))
        c = random.randint(2, 20)
        sum_ab = target_approx // c
        a = random.randint(1, sum_ab - 1) if sum_ab > 2 else 1
        b = sum_ab - a
        
        answer = (a + b) * c
        prompt = f"Give the answer to ({a} + {b}) * {c} and don't say anything else."
        key = ('combo_add_mul', a, b, c)
    
    if answer <= 0 or answer > 100000:
        return None
        
    return prompt, answer, key

def generate_modulo_for_digit(target_digit):
    """Generate modulo operation targeting specific first digit."""
    # Choose modulus and adjust dividend
    modulus = random.randint(10, 99)
    remainder = target_digit
    
    # Make sure remainder < modulus
    if remainder >= modulus:
        remainder = target_digit
        modulus = remainder + random.randint(1, 90)
    
    # Generate dividend that gives this remainder
    multiplier = random.randint(1, 100)
    dividend = modulus * multiplier + remainder
    
    prompt = f"Give the answer to {dividend} mod {modulus} and don't say anything else."
    return prompt, remainder, ('modulo', dividend, modulus)

def generate_simple_for_digit(target_digit, seen_problems):
    """Fallback generator that directly creates numbers with target first digit."""
    # Generate a number with the target first digit
    magnitude = random.randint(0, 3)
    answer = target_digit * (10 ** magnitude) + random.randint(0, 10 ** magnitude - 1)
    
    # Create a simple expression that equals this
    if random.choice([True, False]):
        # Addition
        a = answer // 2
        b = answer - a
        prompt = f"Give the answer to {a} plus {b} and don't say anything else."
    else:
        # Multiplication by 1
        prompt = f"Give the answer to {answer} times 1 and don't say anything else."
    
    return prompt, answer

# ----------------------------
# LLM PROCESSING
# ----------------------------
def get_representations_batch(prompts, task_name, model, tokenizer, batch_size, verbose=False):
    """Process prompts in batches and extract hidden representations."""
    if verbose:
        print(f"\nProcessing {task_name} prompts in batches of {batch_size}...")
    
    layer_reps = {}
    llm_predicted_labels = []
    ground_truth_labels = []
    correctness_flags = []
    total_processed = 0

    if verbose:
        print(f"\nDETAILED {task_name.upper()} PROCESSING:")
        print("=" * 80)

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {task_name} batches", disable=not verbose):
        batch_prompts = prompts[i:i+batch_size]
        batch_texts = [prompt[0] for prompt in batch_prompts]
        batch_answers = [prompt[1] for prompt in batch_prompts]
        batch_gt_digits = [prompt[2] for prompt in batch_prompts]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
                return_dict_in_generate=True,
                output_hidden_states=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        for j, (prompt_text, correct_answer, gt_digit) in enumerate(batch_prompts):
            try:
                # Find the token position that predicts the first digit
                input_length = len(inputs.input_ids[j])
                first_digit_pred_pos, generation_step = find_first_digit_token_position(
                    outputs.sequences[j], tokenizer, input_length
                )
                
                # Get the response text for analysis
                full_text = tokenizer.decode(outputs.sequences[j], skip_special_tokens=True)
                gen_text = full_text[len(prompt_text):].strip()
                
                # Use improved parsing
                extracted_answer = extract_number_from_response(gen_text)
                llm_predicted_digit = get_first_digit(gen_text) if gen_text else None
                if llm_predicted_digit is None:
                    llm_predicted_digit = random.randint(1, 9)
                
                # Check correctness using extracted answer
                is_correct = (extracted_answer == correct_answer)
                
                # Detailed logging for small runs
                if verbose and total_processed < 20:  # Only show first 20 examples in detail
                    print(f"Example {total_processed + 1}:")
                    print(f"  Prompt: {prompt_text}")
                    print(f"  Correct Answer: {correct_answer}")
                    print(f"  LLM Response: '{gen_text}'")
                    print(f"  Extracted Answer: {extracted_answer}")
                    print(f"  Exact Match: {is_correct}")
                    print(f"  Expected First Digit: {gt_digit}")
                    print(f"  LLM First Digit: {llm_predicted_digit}")
                    print(f"  First Digit Correct: {llm_predicted_digit == gt_digit}")
                    print(f"  First Digit Pred Position: {first_digit_pred_pos} (Generation Step: {generation_step})")
                    print()
                
                # Get hidden states from the appropriate generation step
                generation_step = max(0, min(generation_step, len(outputs.hidden_states) - 1))
                hidden_states = outputs.hidden_states[generation_step]
                
                # Get position within the sequence at this generation step
                if generation_step == 0:
                    # Using input positions
                    pos_in_sequence = first_digit_pred_pos
                else:
                    # For later generation steps, use the last position
                    pos_in_sequence = hidden_states[0].shape[1] - 1
                
                for layer_idx, layer_tensor in enumerate(hidden_states):
                    vec = layer_tensor[j, pos_in_sequence, :].cpu().numpy()
                    layer_reps.setdefault(layer_idx, []).append(vec)

                llm_predicted_labels.append(llm_predicted_digit)
                ground_truth_labels.append(gt_digit)
                correctness_flags.append(is_correct)
                total_processed += 1
                
            except Exception as e:
                if verbose:
                    print(f"Error processing example {j} in batch {i//batch_size}: {e}")
                llm_predicted_digit = random.randint(1, 9)
                
                # Use fallback representation from last input position
                try:
                    input_length = len(inputs.input_ids[j])
                    hidden_states = outputs.hidden_states[0]
                    
                    for layer_idx, layer_tensor in enumerate(hidden_states):
                        vec = layer_tensor[j, input_length - 1, :].cpu().numpy()
                        layer_reps.setdefault(layer_idx, []).append(vec)
                except:
                    # Ultimate fallback: random vectors
                    for layer_idx in range(len(outputs.hidden_states[0])):
                        if layer_idx in layer_reps and len(layer_reps[layer_idx]) > 0:
                            vec_dim = layer_reps[layer_idx][0].shape[0]
                            random_vec = np.random.normal(0, 1, vec_dim)
                            layer_reps.setdefault(layer_idx, []).append(random_vec)
                
                llm_predicted_labels.append(llm_predicted_digit)
                ground_truth_labels.append(gt_digit)
                correctness_flags.append(False)
                total_processed += 1

    if verbose:
        print(f"Processed {total_processed}/{len(prompts)} {task_name} examples.")
    
    correct_count = sum(correctness_flags)
    accuracy = correct_count / len(correctness_flags) if correctness_flags else 0
    if verbose:
        print(f"{task_name} LLM Exact Accuracy: {accuracy:.3f} ({correct_count}/{len(correctness_flags)})")
    
    # Calculate first digit accuracy
    first_digit_correct = sum(1 for pred, gt in zip(llm_predicted_labels, ground_truth_labels) if pred == gt)
    first_digit_accuracy = first_digit_correct / len(ground_truth_labels) if ground_truth_labels else 0
    if verbose:
        print(f"{task_name} LLM First Digit Accuracy: {first_digit_accuracy:.3f} ({first_digit_correct}/{len(ground_truth_labels)})")
    
    # Print distribution of ground truth labels
    digit_counts = defaultdict(int)
    for digit in ground_truth_labels:
        digit_counts[digit] += 1
    if verbose:
        print(f"{task_name} digit distribution: {dict(sorted(digit_counts.items()))}")
    
    return layer_reps, llm_predicted_labels, ground_truth_labels, correctness_flags

def generate_and_save_data(model_name, num_prompts, output_dir, batch_size=32, verbose=False, overwrite=False):
    """Main function to generate and save all data."""
    
    # Set up directories
    if overwrite and os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(f"{output_dir}/linear", exist_ok=True)
    os.makedirs(f"{output_dir}/nonlinear", exist_ok=True)
    
    if verbose:
        print("="*50)
        print("MATHEMATICAL INTERPRETABILITY DATA PREPARATION")
        print(f"EXPERIMENT: {os.path.basename(output_dir)}")
        print("="*50)
        print("Generating balanced datasets...")
        print(f"Target: {num_prompts:,} total, ~{num_prompts//9:,} per digit")
    
    # Generate prompts
    linear_prompts = generate_linear_balanced(num_prompts, verbose=verbose)
    nonlinear_prompts = generate_nonlinear_balanced(num_prompts, verbose=verbose)

    if verbose:
        print(f"\nGenerated {len(linear_prompts)} linear prompts")
        print(f"Generated {len(nonlinear_prompts)} non-linear prompts")

    # Load model
    if verbose:
        print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    if verbose:
        print("Model loaded successfully!")

    # Get representations
    linear_data, linear_labels, linear_gt_labels, linear_correctness = get_representations_batch(
        linear_prompts, "linear", model, tokenizer, batch_size, verbose=verbose)
    nonlinear_data, nonlinear_labels, nonlinear_gt_labels, nonlinear_correctness = get_representations_batch(
        nonlinear_prompts, "non-linear", model, tokenizer, batch_size, verbose=verbose)

    # Save data
    if verbose:
        print("\nSaving data...")
    for layer_idx, reps in linear_data.items():
        np.save(f"{output_dir}/linear/X_layer_{layer_idx}.npy", np.array(reps))
    np.save(f"{output_dir}/linear/labels.npy", np.array(linear_labels))
    np.save(f"{output_dir}/linear/gt_labels.npy", np.array(linear_gt_labels))
    np.save(f"{output_dir}/linear/correctness.npy", np.array(linear_correctness))

    for layer_idx, reps in nonlinear_data.items():
        np.save(f"{output_dir}/nonlinear/X_layer_{layer_idx}.npy", np.array(reps))
    np.save(f"{output_dir}/nonlinear/labels.npy", np.array(nonlinear_labels))
    np.save(f"{output_dir}/nonlinear/gt_labels.npy", np.array(nonlinear_gt_labels))
    np.save(f"{output_dir}/nonlinear/correctness.npy", np.array(nonlinear_correctness))

    if verbose:
        print("Data saved successfully!")
    
    # Calculate individual task accuracies
    linear_exact_correct = sum(linear_correctness)
    linear_total = len(linear_correctness)
    linear_exact_accuracy = linear_exact_correct / linear_total if linear_total > 0 else 0
    
    nonlinear_exact_correct = sum(nonlinear_correctness)
    nonlinear_total = len(nonlinear_correctness)
    nonlinear_exact_accuracy = nonlinear_exact_correct / nonlinear_total if nonlinear_total > 0 else 0
    
    linear_first_digit_correct = sum(1 for pred, gt in zip(linear_labels, linear_gt_labels) if pred == gt)
    linear_first_digit_accuracy = linear_first_digit_correct / linear_total if linear_total > 0 else 0
    
    nonlinear_first_digit_correct = sum(1 for pred, gt in zip(nonlinear_labels, nonlinear_gt_labels) if pred == gt)
    nonlinear_first_digit_accuracy = nonlinear_first_digit_correct / nonlinear_total if nonlinear_total > 0 else 0
    
    # Summary
    total_exact_correct = linear_exact_correct + nonlinear_exact_correct
    total_examples = linear_total + nonlinear_total
    overall_exact_accuracy = total_exact_correct / total_examples if total_examples > 0 else 0
    
    total_first_digit_correct = linear_first_digit_correct + nonlinear_first_digit_correct
    overall_first_digit_accuracy = total_first_digit_correct / total_examples if total_examples > 0 else 0
    
    if verbose:
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        print(f"Total Examples: {total_examples}")
        print()
        print("EXACT ACCURACIES:")
        print(f"  Linear:     {linear_exact_accuracy:.3f} ({linear_exact_correct}/{linear_total})")
        print(f"  Non-linear: {nonlinear_exact_accuracy:.3f} ({nonlinear_exact_correct}/{nonlinear_total})")
        print(f"  Overall:    {overall_exact_accuracy:.3f} ({total_exact_correct}/{total_examples})")
        print()
        print("FIRST DIGIT ACCURACIES:")
        print(f"  Linear:     {linear_first_digit_accuracy:.3f} ({linear_first_digit_correct}/{linear_total})")
        print(f"  Non-linear: {nonlinear_first_digit_accuracy:.3f} ({nonlinear_first_digit_correct}/{nonlinear_total})")
        print(f"  Overall:    {overall_first_digit_accuracy:.3f} ({total_first_digit_correct}/{total_examples})")
        
        if overall_exact_accuracy >= 0.9:
            print("\n✅ EXCELLENT! LLM accuracy is 90%+ - ready for full experiment!")
        elif overall_exact_accuracy >= 0.8:
            print("\n⚠️  Good accuracy (80%+) but could be improved")
        else:
            print("\n❌ Low accuracy - consider adjusting prompts or model")
    
    return overall_exact_accuracy, overall_first_digit_accuracy

# Set up seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == "__main__":

    """
    32B
    EXACT ACCURACIES:
      Linear:     0.882 (127/144)
      Non-linear: 0.792 (114/144)  
    """
    
    
    # Test configuration - small scale with detailed output
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
    TEST_NUM_PROMPTS = 512  # 2 examples per digit (1-9)
    TEST_BATCH_SIZE = 64
    
    if MODEL_NAME == "Qwen/Qwen2.5-32B-Instruct":
        temp_save_model_name = 'qwen_32b_it'
    
    EXPERIMENT_NAME = f"{temp_save_model_name}_{TEST_NUM_PROMPTS}_test"
    OUTPUT_DIR = f"../data/math/{EXPERIMENT_NAME}"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    set_seeds(42)
    
    # Run test with detailed output
    exact_acc, first_digit_acc = generate_and_save_data(
        model_name=MODEL_NAME,
        num_prompts=TEST_NUM_PROMPTS,
        output_dir=OUTPUT_DIR,
        batch_size=TEST_BATCH_SIZE,
        verbose=True,  # Detailed output for testing
        overwrite=True  # Overwrite existing data
    )
    
    print(f"\n🎯 TEST COMPLETE!")
    print(f"To run full experiment with 100k examples, use:")
    print(f"generate_and_save_data('{MODEL_NAME}', 100000, '../data/math/qwen_7b_it_100000_balanced', 32, verbose=False)")

    