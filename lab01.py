#1st Question
def pair_sum(arr):
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == 10:
                count += 1
    return count

#2nd Question
def calculate_range(arr):
    if len(arr) < 3:
        return "Range determination not possible"
    
    max_val = arr[0]
    min_val = arr[0]
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
        if arr[i] < min_val:
            min_val = arr[i]
    return max_val - min_val

#3rd Question
def matmul(A,B):
    n=len(A)
    prod=[]
    for i in range(n):
        row=[0]*n
        prod.append(row)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                prod[i][j] += A[i][k] * B[k][j]
    return prod

def matpow(A,m):
    B=A
    n=len(A)
    ans=ans = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    """for i in range(n):
        row=[]
        for j in range(n):
            row.append(1)
        ans.append(row)"""
    while m>0:
        if m%2==1:
            ans=matmul(ans,B)
        B=matmul(B,B)
        m=m//2
    return ans

#4th Question
def occ(word):
    done = {}
    for i in word:
        if i in done:
            done[i] += 1
        else:
            done[i] = 1
    maxchar = None
    maxcount = 0
    for char, count in done.items():
        if count > maxcount:
            maxcount = count
            maxchar = char
    return maxchar, maxcount


def main():
    arr = [2,7,4,1,3,6]
    arr1 = [5,3,8,1,0,4]
    arr2=[1,0,3]
    A=[[1,2,3],[1,2,3],[1,2,3]]
    m=3

    
    count1 = pair_sum(arr)
    range1 = calculate_range(arr1)
    range2 = calculate_range(arr2)
    answer=matpow(A,m)
    
    print("Question 1")
    print("Count of pairs with sum=10:", count1)
    print("Question 2")
    print("Range of arr:", range1)
    print("Range of arr2:", range2)
    print("Question 3")
    print(answer)
    s = "hippopotamus"
    char, count = occ(s)
    print("The highest occurring character is", char, "with a count of", count)
    
if __name__ == "__main__":
    main()
