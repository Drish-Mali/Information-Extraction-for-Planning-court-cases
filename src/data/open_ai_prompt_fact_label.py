import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
from openai import OpenAI
from time import perf_counter
import time
import pandas as pd
from tqdm import tqdm
def truncate_text(text, max_chars=2000):
    if len(text) > max_chars:
        return text[:max_chars]
    else:
        return text
class DataBot:
    def __init__(self, model="gpt-3.5-turbo-0125"):
        self.model = model
        self.bot_name = "DataBot"

    def send_messages(self, message_logs):
       #Add the openai key here
        OPENAI_API_KEY = ""
        
        client = OpenAI(api_key=OPENAI_API_KEY)
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[message_logs[0]]
            )
            

            for choice in response.choices:
                if "text" in choice:
                    return choice.text

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in send_messages: {e}")
            return None

    def get_iob_data(self, df):
        final_answers=[]
        output_df = pd.DataFrame(columns=['paragraph', 'output'])

        for i in tqdm(range(len(df))):
            #print(i)
            #answer = df['filtered_paragraph'][i]
            paragraph=df['paragraph'][i]
            data_message_log = [
                {
                    "role": "system",
                    "content": f"""**Task:** For the given text and determine whether it contains a fact content. Assign the label 1 only when you are extremely certain that the fact content is present (as mostly don't have fact content); otherwise, assign the label 0. You must generate an label for every given input and ouput the label only. The description of text with fact content is delimited by << >>.
            <<
                Fact content: Text containing rules, facts like section (like section 10, s 10 ,S10), Act, article, Amendment, rule, policy,local plan, paragrapgh from NPPF, CPR (civil preceding rule) etc.

            >>
                The few shot examples are delimited by ``` ```
            ```
                1) Input text : For all the above reasons, these second applications to reconsider must fail. They fail to meet any – let alone all – of the criteria set out in CPR 52.30. These are not exceptional cases. There has been no injustice to the applicant. There is no probability of a different result. There was never any tenable basis for an appeal, for the reasons given by both the judge and Lewison L.J.. We consider that neither application for reconsideration was justifiable. The applications before us are therefore dismissed.

                Output: 1


                2) Input text: In Lawal v Circle 33 Housing Trust [2014] EWCA Civ 1514, [2015] 1 P. & C.R. 12, Sir TerenceEtherton, then the Chancellor of the High Court, said at paragraph 65 that the paradigm case for reopening “is where the litigation process has been corrupted, such as by fraud or bias or where the judge read the wrong papers”. He reiterated that the broad principle was that “for an appeal to be reopened, the injustice that would be perpetrated if the appeal is not reopened must be so grave as to overbear the pressing claim of finality in litigation”. Finally, he said:
                “It also follows that the fact that a wrong result was reached earlier, or that there is fresh evidence, or that the amounts in issue are very large, or that the point in issue is very important to one or more of the parties or is of general importance is not of itself sufficient to displace the fundamental public importance of the need for finality.”

                Output: 0

                3) Input text : These and other statements of principle were brought together in the judgment of this court inGoring-on-Thames Parish Council, to which we have already referred. Importantly, at paragraph 15,emphasis was placed on the requirement that “there must be a powerful probability that the decision in question would have been different if the integrity of the earlier proceedings had not been critically undermined”. More recently, the scope of the jurisdiction was summarised byHickinbottom L.J.in Balwinder Singh v Secretary of State for the Home Department [2019] EWCA Civ 1504, paragraph 3, in terms with which we entirely agree:“This is an exceptional jurisdiction, to be exercised rarely: “the injustice that would be perpetrated if the appeal is not reopened must be so grave as to overbear the pressing claim of finality in litigation”(Lawal v Circle 33 Housing Trust [2014] EWCA Civ 1514; [2015] HLR 9 at [65] per Sir TerenceEtherton VC (as he then was)). The jurisdiction will therefore not be exercised simply because the determination was wrong, but only where it can be demonstrated that the integrity of the earlier proceedings has been “critically undermined” (R (Goring-on-Thames Parish Council) v SouthOxfordshire District Council [2018] EWCA Civ 860; [2018] 1 WLR 5161 at [10]-[11].”

                Output: 0

                4) Input : There were several delays prior to the Secretary of State’s decision owing to additional consultations,which included a consultation on the Court of Appeal’s decision in East Northamptonshire DistrictCouncil v Secretary of State for Communities and Local Government (the Barnwell Manor case)[2014] EWCA Civ 137. The Court of Appeal interpreted section 66(1) of the Planning (Listed Buildings And Conservation Areas) Act 1990 as requiring the decision-maker to give “the desirability of preserving the [relevant] building or its setting” not merely careful consideration but considerable importance and weight when balancing the advantages of the proposed development against the harmit might do.
                Output: 1

                5) Input: For these reasons, which are somewhat different from those of the judge, I would dismiss this appeal. I agree.
                Output: 0
            ```
            **Note the output should be 1 or 0. So return only 0 or 1 accordingly**

    The legal text is  {paragraph}"""
                }
            ]
            final_answer = self.send_messages(data_message_log)
            #lines = [line.strip() for line in final_answer.split('\n') if line.strip()]
            # print("*******")
            # print(final_answer)
    
                # Write each line to the TSV file
                # for line in lines:
                #     tsvfile.write(line.replace(' ', '\t') + '\n')
                # # Write an empty line to separate documents
                # tsvfile.write('\n')
            # final_answers.append(final_answer)
            # Save to DataFrame after each iteration
            temp_df = pd.DataFrame({'paragraph': [paragraph], 'output': [final_answer]})
            output_df = pd.concat([output_df, temp_df], ignore_index=True)
            output_df.to_csv('../../data/output_data_fact.csv', index=False) 
        # return final_answers

    def generate_responses_parallel(self):
        df = pd.read_csv("../../data/fact.csv")
        #df['filtered_paragraph'] = df['paragraph'].apply(truncate_text)
        df.reset_index(drop=True, inplace=True)
        self.get_iob_data(df)
        # return iob_data