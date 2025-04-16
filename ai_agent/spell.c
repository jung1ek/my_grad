#include <stdio.h>
#include <ctype.h>

int main()
{
    int N = 26;
    char word[4] = "Zbg";
    int idx = (toupper(word[0]) - 'A') % N;
    printf("%i", idx);
    return 0;
}