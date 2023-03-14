#include <stdio.h>
#include <stdlib.h>

int DIM = 20; // размерность
// комплексное число
struct cuComplex
{
	float r;
	float i;
};
// квадрат модуля комплексного числа
float magnitude(struct cuComplex* a)
{
	return (a->r * a->r + a->i * a->i);
}
// перемножение двух комплексных чисел
void multComplex(const struct cuComplex* a, const struct cuComplex* b, struct
	cuComplex* rezult)
{
	rezult->r = b->r * a->r - b->i * a->i;
	rezult->i = b->i * a->r + b->r * a->i;
}
// сумма двух комплексных чисел
void addComplex(const struct cuComplex* a, const struct cuComplex* b, struct
	cuComplex* rezult)
{
	rezult->r = b->r + a->r;
	rezult->i = b->i + a->i;
}

// алгоритм нахождения точки
int julia(int x, int y)
{
	const float scale = 1.5;
	// центрируем изображение
	float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
	float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);
	// изменяя значения Re(c) и Im(c) можно менять вид фрактала
	struct cuComplex c = { .r = (float)(-0.8), .i = (float)0.156 };
	struct cuComplex a = { .r = jx, .i = jy };
	struct cuComplex temp = { .r = jx, .i = jy };
	// Проверяем быстро ли растет значение |a|*|a|, где a = a*a + c в точке
	//(x, y).
		// В данном случае, проверяется превзойдет ли |a|*|a| значения в одну тысячу,
		//спустя двести итерации.
		// Изменяя значения 200 и 1000, вы измените количесто отображаемых точек
		for (int i = 0; i < 200; ++i)
		{
			multComplex(&a, &a, &temp);
			a.r = temp.r;
			a.i = temp.i;
			addComplex(&a, &c, &a);
			// если амплитуда растет
			if (magnitude(&a) > 1000)
				return 0;
		}
	return 1;
}
// главная функция
void kernel(unsigned char* ptr)
{
	for (int y = 0; y < DIM; ++y)
	{
		for (int x = 0; x < DIM; ++x)
		{
			int offset = x + y * DIM;
			int juliaValue = julia(x, y);
			ptr[offset] = ' ';
			ptr[offset] = ' ' + juliaValue * 10;
		}
	}
}

int main()
{
	// размер сетки
	int size = DIM * DIM;
	// выделение памяти под массив символов
	unsigned char* ptr = (unsigned char*)malloc(size * sizeof(unsigned char));
	// вызов главной функции
	kernel(ptr);
	// переменная для преобразования двумерной индексации к одномерной
	int offset;
	// цикл вывода изображения в консоль
	for (int y = 0; y < DIM; ++y)
	{
		for (int x = 0; x < DIM; ++x)
		{
			offset = x + y * DIM;
			printf("%c", ptr[offset]);
		}
		printf("\n");
	}
	free(ptr);
	return 0;
}
