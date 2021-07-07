s = open('student.txt', 'r', encoding="UTF8")
A = s.read().split('\n')
s.close()

student, Id = [], []
for i in range (len(A)):
    Id.append(A[i].split('\t')[0]) # if it is splited by 'tab'. else, consider other options.
    student.append(A[i].split('\t')[1])


check = '0525'

nothere = 0
for i in range (len(A)):
    with open(check+'.txt', encoding="UTF8") as f:
        if (student[i] in f.read()):
            print(Id[i], student[i], ': O')
        elif (Id[i] in f.read()):
            print(Id[i], student[i], ': O')
            
        else:
            print(Id[i], student[i], ': X!')
            nothere = nothere + 1
print(f'absent: {nothere} out of {len(A)}')


