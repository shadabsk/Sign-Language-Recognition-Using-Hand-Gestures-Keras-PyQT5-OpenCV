#This is the script that generate random integer & string
fp=open("File2.txt","w") #output file to store console outputs

import random #For random function
import string  #for string uses

a=[] #stores random integer
#string.strs='ABCDEFGHIJKLMNOPQRSTUVWXYZ' #for sear no alphabet generator

#for area of interest generation
#string.strs='COMPUTER ENGINEERING','MECHANICAL ENGINEERIN','CIVIL ENGINEERING','ELECTRICAL ENGINEERING','ENGINEERING','PHARMACY','ARCHITECTURE','DOCTOR','POLICE','TEACHING','BUSSINESS MANAGEMENT','HOTEL MANAGEMENT','MILITRY','AIR FORCE','NAVY','PILOT','CA','BUSSINESSMAN','DATABASE ADMINSITRATOR','SOFTWARE DEVELOPER','SOFTWARE TESTER','WEB DEVELOPER','IS OFFICER','ACCOUNTANT','BANK MANAGER','FASHION DESIGNER','CHEMICAL ENGINEER','ENVIRONMENTAL ENGINEER','GEATECHNICAL ENGINEER','AGRTICULTURAL ENGINEER'

#how no. of record should be generate
n=int(input("Enter number of elements:"))


for j in range(n):
    #a.append(random.choice(string.strs))
    a.append(random.randint(3,98))

print(a) #print random data as list

values = '\n'.join(map(str, a)) #storing list element by seperating new line
print(values) #printing random data to console

#writing random into file
for i in values:
	fp.write(i)
	
fp.close() #closing file after operation	
