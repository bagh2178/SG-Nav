import multiprocessing  
  
def start_child_process(process_id):  
    import subprocess  
    subprocess.Popen(["python", "-m", "start", str(process_id)])  
  
if __name__ == "__main__":  
    processes = []  
    for i in range(11):  
        p = multiprocessing.Process(target=start_child_process, args=(i,))  
        p.start()  
        processes.append(p)  
  
    for p in processes:  
        p.join()
