#include <stdio.h>

int len=0;
//function to print the array
void printarray(int arr[], int size)
{
    int i,j;
    for(i=0; i<size; i++)
    {
        // printf("%d\t",arr[i]);
    }
    len++;
    // printf("%d\n",len);
}

//function to swap the variables
void swap(int *a, int *b)
{
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

//permutation function
void permutation(int *arr, int start, int end)
{
    if(start==end)
    {
        printarray(arr, end+1);
        return;
    }
    int i;
    for(i=start;i<=end;i++)
    {
        //swapping numbers
        swap((arr+i), (arr+start));
        //fixing one first digit
        //and calling permutation on
        //the rest of the digits
        permutation(arr, start+1, end);
        swap((arr+i), (arr+start));
    }
}

int main()
{
    //taking input to the array
    int size=10;
    int arr[size];
    for(int i=0;i<size;i++)
        arr[i]=i+1;
    //calling permutation function
    permutation(arr, 0, size-1);
    printf("%d\n",len);

    return 0;
}
