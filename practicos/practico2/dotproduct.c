
#define N (1 << 20)

float a[N], b[N], c[N];

int main(void) {
    unsigned int i=0;
    for (i=0; i<N; ++i) {
        a[i] = b[i]*c[i];
    }
    return 0;
}
