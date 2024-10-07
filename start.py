import os
import sys  
  
def main(process_id):  
    devices = ['0', '0', '1', '1', '2', '2', '3', '3', '4', '4', '5']

    splits_l = [0, 2, 4, 6, 7, 8, 9, 10]
    splits_r = [2, 4, 6, 7, 8, 9, 10, 11]
    split_l = splits_l[process_id]
    split_r = splits_r[process_id]

    split_l = process_id
    split_r = process_id + 1
    device = devices[process_id]
    print(f"subprocess {process_id}")  
    os.system(f'CUDA_VISIBLE_DEVICES={device} python SG_Nav.py --evaluation local --reasoning both --visulize --split_l {split_l} --split_r {split_r}')
  
if __name__ == "__main__":  
    if len(sys.argv) > 1:  
        process_id = int(sys.argv[1])  
        main(process_id)  

    print('done')
