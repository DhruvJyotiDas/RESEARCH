# Artifact audit of the HF release vikash0132/KG-CoQA (1,000 rows).
# Usage: python3 -I kgcoqa_artifact_audit.py <hf_cache_dir>   (needs: datasets, scikit-learn, numpy)
# Reports: pronoun/length/triple-count rules, story-grouped question-only LR, label/story structure, relation structure.
import sys, re, collections, statistics as st, numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score
ds = load_dataset("vikash0132/KG-CoQA", cache_dir=sys.argv[1])["train"]
rows=list(ds); y=np.array([int(r["is_ambiguous"]) for r in rows]); groups=[r["coqa_story"] for r in rows]
q=[r["question"] for r in rows]
PRON=re.compile(r"\b(he|she|it|they|his|her|its|their|them|him|hers|theirs|this|that|these|those)\b",re.I)
def trip(p): return re.findall(r"\(([^)]*)\)-\[([^\]]*)\]->\(([^)]*)\)",p)
# 1 rules
pr=np.array([int(bool(PRON.search(x))) for x in q]); print("pronoun rule acc",accuracy_score(y,pr),"f1",f1_score(y,pr))
nt=np.array([len(trip(r["based_on_pattern"])) for r in rows]); print("triples>=2 rule acc",accuracy_score(y,(nt>=2).astype(int)))
ln=np.array([len(x.split()) for x in q]); print("len<=6 rule acc",accuracy_score(y,(ln<=6).astype(int)))
# 2 question-only LR, grouped by story
gk=GroupKFold(5)
for name,vec in [("word1-2",TfidfVectorizer(ngram_range=(1,2),min_df=2)),("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(2,5),min_df=2))]:
    X=vec.fit_transform(q); p=cross_val_predict(LogisticRegression(max_iter=2000),X,y,cv=gk,groups=groups)
    print("question-only LR",name,"acc",accuracy_score(y,p),"macroF1",f1_score(y,p,average="macro"))
# 3 label by story pairing
by=collections.defaultdict(list)
for r in rows: by[r["coqa_story"]].append(int(r["is_ambiguous"]))
print("stories with mixed labels",sum(1 for v in by.values() if len(set(v))==2),"all-amb",sum(1 for v in by.values() if set(v)=={1}),"all-non",sum(1 for v in by.values() if set(v)=={0}))
# 4 question string overlap across labels
qa=collections.Counter(r["question"] for r in rows if r["is_ambiguous"]); qn=collections.Counter(r["question"] for r in rows if not r["is_ambiguous"])
print("distinct amb q",len(qa),"of",sum(qa.values()),"; top amb:",qa.most_common(8)); print("distinct non-amb q",len(qn),"; overlap strings",len(set(qa)&set(qn)))
# 5 relation structure in ambiguous rows
same_rel=0; same_rel_n=0; tot=0
for r in rows:
    if r["is_ambiguous"]:
        t=trip(r["based_on_pattern"]); rels={b.lower() for a,b,c in t}; subj={a for a,b,c in t}; obj={c for a,b,c in t}
        tot+=1; same_rel+= (len(rels)==1); 
print("amb rows with a single shared relation",same_rel,"/",tot)
# distinct subjects vs objects: multi-parent?
cnt=collections.Counter()
for r in rows:
    if r["is_ambiguous"]:
        t=trip(r["based_on_pattern"]); subj={a for a,b,c in t}; obj={c for a,b,c in t}
        cnt[("distinct_subj>1" if len(subj)>1 else "1subj")+"/"+("distinct_obj>1" if len(obj)>1 else "1obj")]+=1
print(cnt)
# 6 pronoun type and q wh
print(collections.Counter(re.sub(r"[^a-z ]","",x.lower()) for x in qa.elements()).most_common(0))
# 7 context is verbalised triples? ctx length
print("ctx words amb",st.mean(len(r["context"].split()) for r in rows if r["is_ambiguous"]),"non",st.mean(len(r["context"].split()) for r in rows if not r["is_ambiguous"]))
# 8 non-amb questions with pronoun
ex=[r["question"] for r in rows if not r["is_ambiguous"] and PRON.search(r["question"])][:5]; print("non-amb w/ pron:",ex)
# 9 story length
print("story words",st.mean(len(s.split()) for s in set(groups)))
# 10 are non-amb question entity names present in context (named)
