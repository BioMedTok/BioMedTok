import os
import re

from tqdm import tqdm
from pqdm.processes import pqdm

f_in = open("./morpheme_fr_struct.txt","r")
morphemes = [s.lower() for s in f_in.read().split("\n") if len(s) > 3]
morphemes = list(reversed(sorted(morphemes, key=len)))
f_in.close()

NBR_THREADS = 10

print(morphemes)

def process(lines):

    local_res = []
    
    for line in tqdm(lines):

        if len(line) <= 0:
            continue

        # print(line)
        
        # Split into words
        # words = "Pourquoi pas, aussi bien la céphalographie ou description de tête, la podographie ou description de pied".lower().split(" ")

        line = line.replace("\n"," ").lower()

        # Add identifiers and boolean for morpheme for each words
        words = [(w_idx, w) for w_idx, w in enumerate(line.split(" "))]

        # Split the words into subwords until no morphemes from the list was found
        new_words = []

        # For each word
        for w_idx, w in words:

            # Add surrounding spaces to consider it as a subword unit
            current_word_subtokens = " " + w + " "

            # Should continue ?
            trigger_bool = True

            # Try to find morpheme in the remaining text

            # While all the phonemes in the word haven't been found then continue
            while trigger_bool:

                # Set to stop
                trigger_bool = False

                # Split all the subwords units obtained through the time
                for subword in current_word_subtokens.split(" "):

                    # If the current subword contains |True, its mean that it is a morpheme which doesn't need to be splitted again
                    if "|True" in subword or len(subword) <= 0 or subword == " ":
                        continue
                    
                    # print(f"# {subword} #")

                    # If its not a morpheme, look a the list of morphemes and check if any of them can fit in the current word
                    for m in morphemes:

                        # if m == "céphal":
                        #     print(f"--- {m} & {subword} => {m in subword} ---")
                        
                        # If its contained in it
                        if m in subword:

                            # Add spaces
                            old_t = f" {subword} "

                            # Separate the morpheme from the others subword units
                            new_t = " " + subword.replace(m, f" {m}|True ") + " "
                            current_word_subtokens = current_word_subtokens.replace(old_t, new_t)
                            # print(f"--- {current_word_subtokens} ---")                   

                            # Set to continue since we have found another morpheme
                            trigger_bool = True

                            # Go to the next subword
                            break

            # The end sequence of tokens remove the |True tags and dead spaces
            # new_words.extend([c.replace("|True","") for c in current_word_subtokens.split(" ") if len(c) > 0])

            # The end sequence of tokens without including any |True subword and dead spaces
            new_words.extend([c for c in current_word_subtokens.split(" ") if len(c) > 0 and "|True" not in c])
            
            # print(new_words)

        # print("*"*50)
        # print(words)
        # print("*"*50)
        # print(new_words)

        local_res.append(new_words)
    
    return local_res
    # exit(0)

for data_file in os.listdir("./sources/"):

    data_path = f"./sources/{data_file}"
    # new_data_path = f"./sources_preprocessed/{data_file}_morphemes-included"
    new_data_path = f"./sources_preprocessed/{data_file}_morphemes-excluded"

    with open(data_path) as fp:

        # for line in tqdm(lines):
        lines = fp.readlines()

        n_elements = int(len(lines) / NBR_THREADS)

        chunks = [lines[x:x+n_elements] for x in range(0, len(lines), n_elements)]
        
        # print(len(chunks))
        # print(type(chunks))
        # print("*"*50)
        
        # print(len(chunks[0]))
        # print(type(chunks[0]))
        
        # exit(0)

        # response = process(chunks[0])
        # print(response)
        # exit(0)

        thread_res = pqdm([[c] for c in chunks], process, n_jobs=NBR_THREADS, argument_type='args')
        # print("--- thread_res start")
        # print(thread_res)
        # print("--- thread_res end")

        print(data_path)
        print(new_data_path)
        print()
        f_out = open(new_data_path, "w")

        for thread_lines in thread_res:
            # print(thread_lines)
            for current_line in thread_lines:
                f_out.write(" ".join(current_line) + "\n")
        f_out.close()
