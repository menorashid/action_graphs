import random

def writeFile(file_name,list_to_write):
    with open(file_name,'wb') as f:
        for string in list_to_write:
            f.write(string+'\n');

def readLinesFromFile(file_name):
    with open(file_name,'rb') as f:
        lines=f.readlines();
    lines=[line.strip('\n') for line in lines];
    return lines

def writeSplitFiles(input_file,split_sizes,out_files):
	assert sum(split_sizes)==1.
	assert len(split_sizes)==len(out_files)

	lines = readLinesFromFile(input_file)
	random.shuffle(lines)
	
	split_sizes = [int(round(len(lines)*val)) for val in split_sizes]
	diff = len(lines) - sum(split_sizes)
	if diff!=0:
		split_sizes[0]= split_sizes[0]+diff # get rid of rounding errors 
	assert len(lines) == sum(split_sizes)

	split_idx = [0]+[sum(split_sizes[:idx]) for idx in range(1,len(split_sizes)+1)]
	
	for idx_out_file, out_file in enumerate(out_files):
		lines_to_write = lines[split_idx[idx_out_file]:split_idx[idx_out_file+1]]
		print 'writing',out_file,'of size',len(lines_to_write)
		writeFile(out_file,lines_to_write)


def main():
	input_file = 'all_data.txt'
	split_sizes = [0.8,0.1,0.1]
	out_files = ['train.txt','val.txt','test.txt']
	writeSplitFiles(input_file,split_sizes,out_files)


if __name__=='__main__':
	main()