#include <cudaThreadSingleton.h>
#include <cstdlib>

template<class T>
T* cplThreadSingleton<T>::insA[MAX_NUMBER_GPU];

template<class T>
cplThreadSingleton<T>::cplThreadSingleton(){
    for (int i=0; i< MAX_NUMBER_GPU; ++i)
        insA[i] = NULL;
} // ctor hidden

template<class T>
T& cplThreadSingleton<T>::Instance(int id) {
    if (!insA[id])
        if (true){
            // Need scope lock
            if (!insA[id])    {
                insA[id] = new T;
            }
        }
    return *insA[id];
}

template<class T>
void cplThreadSingleton<T>::DestroyInstance(int id) {
    if (insA[id]) {
        if (true){
            // Need scope lock here 
            if (insA[id]) {
                delete insA[id];
                insA[id] = NULL;
            }
        }
    }
}

#include <cudaReduce.h>
template class cplThreadSingleton<cplReduce>;




