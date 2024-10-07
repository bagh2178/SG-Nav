




from asyncore import write


results_list = []

        
with open('output_new/FBE_consist/results_p1.txt') as f:
    for line in f.readlines():
        results_list.append(line)
        
# with open('output_new/FBE_consist/results_p1_cont.txt') as f:
#     for line in f.readlines():
#         results_list.append(line)    

with open('output_new/FBE_consist/results_p2.txt') as f:
    for line in f.readlines():
        results_list.append(line)
        
    
"""
with open('output/UNIQA/results_500_p2_cont1.txt') as f:
    for line in f.readlines():
        results_list.append(line)
        
with open('output/UNIQA/results_500_p1.txt') as f:
    for line in f.readlines():
        results_list.append(line)
        
with open('output/UNIQA/results_500_p1_cont.txt') as f:
    for line in f.readlines():
        results_list.append(line)


with open('output/UNIQA/results_500_p1_cont1.txt') as f:
    for line in f.readlines():
        results_list.append(line)

with open('output/UNIQA/results_500_p1_cont2.txt') as f:
    for line in f.readlines():
        results_list.append(line)
        
with open('output/UNIQA/results_500_p1_cont3.txt') as f:
    for line in f.readlines():
        results_list.append(line) 
"""  
result_set = set(results_list)

with open('output_new/FBE_consist/result_metrics_gather_500.txt','w') as f:
    for result in result_set:
        f.write(result)
results_list = []
with open('output_new/FBE_consist/result_metrics_gather_500.txt','r') as f:
    for line in f.readlines():
        results_list.append(line)

spl_list = []
sr_list = []
softspl_list = []
for result in results_list:
    later = result.split("ce_to_goal': ")[-1]
    dist = float(later.split(',')[0])
    later = result.split("'success': ")[-1]
    success = float(later.split(',')[0])
    if dist <=0.1:
        sr_list.append(1.)
    else:
        sr_list.append(0)
    
    later = result.split("'softspl': ")[-1]
    softspl_list.append(float(later.split('}')[0]))
    
    later = result.split("'spl': ")[-1]
    if dist <=0.1:
        spl_list.append(float(later.split(',')[0]))
    else:
        spl_list.append(0)
    
print(sum(spl_list)/len(spl_list))
print(sum(sr_list)/len(spl_list)) 
print(sum(softspl_list)/len(spl_list)) 
print(len(spl_list))

"""
spl_list = []
for result in results_list:
    later = result.split("'spl': ")[-1]
    spl_list.append(float(later.split(',')[0]))

sr_list = []
for result in results_list:
    later = result.split("'success': ")[-1]
    sr_list.append(float(later.split(',')[0]))

softspl_list = []
for result in results_list:
    later = result.split("'softspl': ")[-1]
    softspl_list.append(float(later.split('}')[0]))
    
print(sum(spl_list)/len(spl_list))
print(sum(sr_list)/len(spl_list)) 
print(sum(softspl_list)/len(spl_list)) 
"""