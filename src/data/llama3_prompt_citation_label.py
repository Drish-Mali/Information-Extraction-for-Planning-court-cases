import json
import torch
import pandas as pd
from tqdm import tqdm
import re
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          BitsAndBytesConfig,
                          pipeline)
from accelerate import Accelerator  # Correct import for Accelerator
import datasets
from torch.utils.data import DataLoader, Dataset
class CitationDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx]['paragraph']
HF_TOKEN = ""
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def get_response():
    # Setup Accelerator for optimized multi-GPU usage
    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Let Accelerate handle device mapping
        quantization_config=bnb_config,
        token=HF_TOKEN
    )
    
    # Move model to accelerator
    

    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=40,
        temperature=0.1,
        device_map="auto"
    )

    # Stream the dataset for better memory management
    df = pd.read_csv("./data/new_main_text.csv")

    facts = []
    
    dataset = CitationDataset(df)
    dataloader = DataLoader(dataset, batch_size=25, shuffle=False)
    model,dataloader = accelerator.prepare(model,dataloader)
    for batch_texts in tqdm(dataloader, total=len(dataloader)):
        for text in batch_texts:   
            if len(text) > 10000:
                text=text[:10000]     
            system_prompt = f""" [INST]
                Task:  For the given text and determine whether it contains a fact content. Assign the label 1 only when you are extremely certain that the fact content is present; otherwise, assign the label 0. You must generate an label for every given input. The description of text with fact content is delimited by << >>.
                <<
                    Fact content: Text containing rules, facts like section (like section 10, s 10 ,S10), Act, article, Amendment, rule, policy,local plan, paragraph from NPPF, CPR (civil preceding rule) etc.
                
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
                    

                    6) Input : 82.   The Report contained a like for like comparison of four sites, CA5, FX3, GH11 and OC5. In Chapter 5 “Constraints and Opportunities” FX3 was assessed for accessibility by rail as follows:-      “there is currently no direct access to the Leeds-HarrogateYork line. As the rail line runs to the south west of the site there may be the potential to develop a new station stop, preferably to the north of the A59 so as to be within walking/cycling distance of the majority of the site. The former Goldsborough station site lies to the southwest of the site although outside of the site boundary shown in the draft Local Plan. The development promoter has undertaken initial investigations on the feasibility of reopening a station in this location to serve the new settlement. This could be a potentially complex solution and without certainty: as it currently stands, a station and rail service are not in place. Knaresborough and Cattal rail stations, the nearest existing stations, are outside of walking distances but potentially accessible by improved bus services.”      By contrast the report’s appraisal of GH11 on this subject was:-      “the site benefits from two operational stations within walking distance of the whole site offering choice.”

                    Output: 0
                    
                    7) Input: Policy 13 of the WCS forms part of the Council’s development strategy under the
                    section titled “Sustainable Environment”, and it contributes to the strategic objectives
                    set out in paragraphs 8.1 and 8.2 of the explanatory text, including conserving and
                    enhancing the natural environment. Policy 13 provides, so far as is material. The case is dissmissed.
                    
                    Output: 1
                    8) Input: Both the Supreme Court and the Court of Appeal have, inrecent cases, emphasized the limits to the court's role inconstruing planning policy (see the judgment of Lord Carnwathin Suffolk Coastal District Council v Hopkins Homes Ltd.[2017] UKSC 37, at paragraphs 22 to 26, and my judgment inMansell v Tonbridge and Malling Borough Council [2017]EWCA Civ 1314, at paragraph 41). More broadly, though inthe same vein, this court has cautioned against the dangers ofexcessive legalism infecting the planning system – a warning Ithink we must now repeat in this appeal (see my judgment inBarwood Strategic Land II LLP v East Staffordshire BoroughCouncil [2017] EWCA Civ 893, at paragraph 50). There is noplace in challenges to planning decisions for the kind ofhypercritical scrutiny that this court has always rejected –whether of decision letters of the Secretary of State and hisinspectors or of planning officers' reports to committee. Theconclusions in an inspector's report or decision letter, or in anofficer's report, should not be laboriously dissected in an effortto find fault (see my judgment in Mansell, at paragraphs 41 and42, and the judgment of the Chancellor of the High Court, atparagraph 63).”
                    Output: 0
                    
                    9) Input :  A local planning authority may be substantially prejudiced by a decision to grant permission where the planning considerations on which the decision is based, particularly if they relate to planning policy, are not explained sufficiently clearly to indicate what, if any, impact they may have on future applications (per Lord Brown in South Bucks, at [30], citing the judgment of Lord Bridge in Save Britain’s Heritage v Number 1 Poultry Ltd [1991] 1 WLR 153.
                    Output: 0
                    
                    10) Input: 50. Refusal of planning permission on grounds of prematurity
                        will seldom be justified where a draft plan has yet to be
                        submitted for examination; or - in the case of a neighbourhood
                        plan - before the end of the local planning authority publicity
                        period on the draft plan. Where planning permission is refused
                        on grounds of prematurity, the local planning authority will
                        need to indicate clearly how granting permission for the
                        development concerned would prejudice the outcome of the
                        plan-making process.”
                    Output: 0
                    
                    11) Input :26.   He went on to say, under the heading “Planning policy and the location of the proposed development” (in paragraphs 8 to 12):      “8. Applications for planning permission are determined in accordance with the development plan, unless material considerations indicate otherwise. The development plan for the area in which the appeal site is located includes the East Staffordshire Local Plan 2012-2031 (‘Local Plan’). The majority of the appeal site is located within Outwoods Parish with the site access falling with Horninglow and Eton Parish. As a result, the Outwoods Neighbourhood Plan and the Horninglow and Eton Neighbourhood Plan also form part of the development plan in relation to the site.      9. The spatial strategy of the Local Plan encapsulated in Strategic Policy 2 is to focus development within the settlement boundaries in a hierarchy of main towns. Burton upon Trent is at the top of this hierarchy, followed by strategic villages and then local service villages. Strategic Policy 4 identifies housing allocations in the Local Plan for main towns and villages. Development outside the settlement boundaries is strictly controlled by Strategic Policy 8.      10. The appeal site lies next to, but outside, the settlement boundary for Burton upon Trent. As a result, for planning policy purposes it lies within the open countryside where Strategic Policy 8 strictly controls development. As the proposal would not comply with any of the exceptions set out in this policy and the site is not a strategic allocation in the Local Plan the scheme would be contrary to Strategic Polices 2, 4 and 8. In terms of the Neighbourhood Plans, the location of the proposed development would not be contrary to their policies.      11. The National Planning Policy Framework (‘the Framework’) is an important material consideration. Paragraph 14 advises that a presumption in favour of sustainable development lies at the heart of the Framework and paragraph 49 advises that housing applications should be considered in this context. In practice this means that proposals which accord with the development plan should be approved without delay.”
                    Output: 1
                    
                    12) Input : 71.   In July 2015 HBC published the Local Plan: Issues and Options consultation document. The accompanying Draft Sustainability Appraisal: Interim Report assessed 11 growth strategies, from which HBC selected 5 for consultation. Option 5 was for “creating a new settlement within the A1(M) corridor to accommodate up to 3,000 new homes”, with the remaining housing requirement being met in the main urban areas of Harrogate, Knaresborough, Ripon, market towns and villages. The SA referred at p. 206 to an area of search running broadly north/south for about 3 miles either side of the A1(M).
                    Output: 0
                    
                    13) Input: 4.   Practice Direction 8C was inserted into the Civil Procedure Rules by the Civil      Procedure (Amendment No.4) Rules 2015 (81  st Update) which remains on the Civil Procedure Rules section of the Justice website. It is apparent that, notwithstanding the creation of Form N208 PC, Paragraph 2.1 of Practice Direction 8C referred to the prescribed form as “[in] practice form N208”, i.e. the standard Part 8 claim form. As the Court will be aware, this remains the case today. I am aware that, despite the indication given in the Rules, persons seeking to issue statutory review claims to which the permission filter applied in the weeks following 26 October 2015, were instructed to complete the new form N208PC. A stock of the forms was kept in the Administrative Court Office for that purpose.
                    Output: 1
                    
                    14) Input: 2.   The first application was refused by the Council in November 2013 and the second application was refused by the Council in July 2017. Appeals against the refusals were dismissed by the Secretary of State for Housing, Communities and Local Government in November 2018. The appellant brought a claim under  s.288 of the Town and Country Planning Act 1990 to quash the Secretary of State’s decision. On 2     August 2019, Dove J, sitting in the Planning Court, dismissed the appellant’s claim. The appellant now appeals against that dismissal, permission to appeal having been granted by Lewison LJ on 16 December 2019.49.   It is not necessary, therefore, to grapple with other parts of the judge’s analysis, where he engaged with concepts such as the “residual scope for the exercise of discretion” (paragraph 24 of his judgment), “an algorithm to describe the process laid down in paragraph n paragraph [14 of the N and the question?” (paragraph 54ragraph [14 of the NPPF]” (paragraph 26), and the question “How exceptional is exceptional?” (paragraph 54 the NPPF]”.
                    Output: 0
                    
                    15) Input: 11.   The essential facts are not in dispute. Indeed, the Secretary of State has expressly accepted the account of events on 23 and 24 March 2016   set out  in Mr Croke’s letter to the court  dated 26 April 2016 and in the witness statement of Mr James Miller dated 8 June 2016. Both the letter and the witness statement contain a statement of truth
                    Output: 0
                    
                    16) Input : 271.   The NWR Scheme, if it proceeds, will not provide any further airport capacity until 2026 at the earliest, and will deliver that capacity to 2060 and beyond. Like the AC, the Secretary of State thus had to make assessments as to future conditions.
                    Output: 0
                    
                    17) Input : 25.   The interpretation of paragraph 14 of the 2012 NPPF was further considered by Lord Carnwath in  Hopkins Homes . At paragraph 54 to 55 he observed:     “54. The general effect is reasonably clear. In the absence of relevant or up-todate development plan policies, the balance is tilted in favour of the grant of permission, except where the benefits are ‘significantly and demonstrably’ outweighed by the adverse effects or where ‘specific policies’ indicate otherwise. (See also the helpful discussion by Lindblom J in  Bloor Homes ….).     55. It has to be borne in mind also that paragraph 14 is not concerned solely with housing policy. It needs to work for other forms of development covered by the development plan, for example employment or transport. Thus, for example, there may be a relevant policy for the supply of employment land, but it may become out-of-date, perhaps because of the arrival of a major new source of employment in the area. Whether that is so, and with what consequence, is a matter of planning judgment, unrelated of course to paragraph 49 which deals only with housing supply. This may in turn have an effect on other related policies, for example for transport. The pressure for new land may mean in turn that other competing policies will need to be given less weight in accordance with the tilted balance. But again that is a matter of pure planning judgment, not dependent on issues of legal interpretation  .”
                    Output: 1
                    
                    18) Input: 47.  In the Hoplands case it was contended that the decision was “contrary to binding EU law in the appellant’s favour” and that the court had acted ultra vires , because it was obliged to refer the matter to the CJEU. Complaint was made that Lewison L.J. had failed to give reasons for refusing to make such a referral. In the Chislet case similar complaints were made; it was contended that Lewison L.J. had failed to “engage with [the applicant’s] point that CJEU jurisprudence takes a broader approach to the scope of assessment”. It was submitted that the proper approach to EIA in this case has     “simply not been addressed”.   It was asserted  that ground 4 (the attack on the adequacy of the substance of the HRA) was not addressed at all, and an attempt was made to rely on the fresh evidence that Lewison L.J. had refused to permit the applicant  to adduce.
                    Output: 0
                ``` 
                
                
                
                Provide label to the text delimited by $$ $$
                [/INST]
                [INPUT]
                $$ {text} $$
                [/INPUT]    
                        """

                
            sequences = text_generator(system_prompt)
            gen_text = sequences[0]["generated_text"]
            #print("************")
            #print(gen_text)   
       
    df['fact'] = facts
    return df

if __name__ == "__main__":
    response=get_response()
    # print(response.head())
    response.to_csv("./data/csv_files/citation_df.csv",index=False)
