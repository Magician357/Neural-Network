posibilities=[1,2,3,4,5,6,7,8,9]
full=[]
for n in posibilities:
    for i in posibilities:
        full.append(((n,i),tuple([n+i])))
full=tuple(full)