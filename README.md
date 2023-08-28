# One-line distillation from GPT-4 to GPT-3.5, a featherweight library


The release of the GPT-3.5 fine-tuning API opens up the possibility to distill from GPT-4. For a specific task, we could theoretically 
reach similar performance with lower cost, reduced latency, and higher rate limit. I'm still experimenting with task granularity/data quantitity 
to achieve this distillation but putting the code here
in case it is useful to everyone.



### **What it does:**
- Given a list of input prompts
- generate the answers using GPT-4
- upload the file to openAI
- fine-tune a GPT-3.5 model for you. 




### **Configurable parameters:**
- GPT-4 parameters: temperature, max_tokens, system_prompt
- fine-tuning parameters: n_epochs and repetitions



### **Instructions:**
```
pip install -r requirements.txt
python main.py your_file.txt
```





### Todos that I will get to at some point:
- Cost estimation: How much did the distillation cost and at when does fine-tuned GPT-3.5 break even with GPT-4
- Data Augmentation: Augment from seed data using GPT-4
- A prettier loading spinner. I like spinny things, bite me.


*Disclaimer: It is unclear whether commercial usage of distillation is violating the openAI ToS, this library is for research purpose only*
