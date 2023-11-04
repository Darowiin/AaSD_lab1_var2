#include <iostream>
#include <complex>
#include <random>
#include <ctime>

using namespace std;

template <typename T>
class Matrix {
	int _rows;
	int _cols;
	T** _data;
public:
	Matrix(int rows, int cols);
	Matrix(int rows, int cols, const T& lower, const T& upper);
	Matrix(int rows, int cols, complex<T>);
	Matrix(int rows, int cols, const complex<T>& lower, const complex<T>& upper);
	Matrix(const Matrix<T>& other);
	Matrix(const Matrix<complex<T>>& other);
	~Matrix();

	void set_rows(int rows);
	void set_cols(int cols);
	int get_rows() const;
	int get_cols() const;

	T& operator()(int row, int col) const;
	Matrix<T> operator+(const Matrix<T>& other) const;
	Matrix<complex<T>> operator+(const Matrix<complex<T>>& other) const;
	Matrix<T> operator+=(const Matrix<T>& other);
	Matrix<complex<T>> operator+=(const Matrix<complex<T>>& other);
	Matrix<T> operator-(const Matrix<T>& other) const;
	Matrix<complex<T>> operator-(const Matrix<complex<T>>& other) const;
	Matrix<T> operator-=(const Matrix<T>& other);
	Matrix<complex<T>> operator-=(const Matrix<complex<T>>& other);
	Matrix<T> operator*(const Matrix<T>& other) const;
	Matrix<complex<T>> operator*(const Matrix<complex<T>>& other) const;
	Matrix<T> operator*(const T& scalar) const;
	Matrix<complex<T>> operator*(const complex<T>& scalar) const;
	Matrix<T> operator/(const T& scalar) const;
	Matrix<complex<T>> operator/(const complex<T>& scalar) const;

	bool operator==(const Matrix<T>& other) const;
	bool operator==(const Matrix<complex<T>>& other) const;
	bool operator!=(const Matrix<T>& other) const;
	bool operator!=(const Matrix<complex<T>>& other) const;

	T trace() const;

	friend ostream& operator<<(ostream& os, const Matrix<T>& matrix) {
		for (int i = 0; i < matrix._rows; i++) {
			for (int j = 0; j < matrix._cols; j++) {
				os << matrix._data[i][j] << ' ';
			}
			os << '\n';
		}
		return os;
	}
};
template<typename T>
Matrix<T>::Matrix(int rows, int cols) : _rows(rows), _cols(cols) {
	_data = new T * [_rows];
	for (int i = 0; i < _rows; ++i) {
		_data[i] = new T[_cols]();
	}
}
template<typename T>
Matrix<T>::Matrix(int rows, int cols, const T& lower, const T& upper) {
	_rows = rows;
	_cols = cols;
	_data = new T * [_rows];
	for (int i = 0; i < _rows; ++i) {
		_data[i] = new T[_cols];
	}

	srand(static_cast<unsigned>(time(nullptr)));

	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			_data[i][j] = lower + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX) / (upper - lower));
		}
	}
}
template<typename T>
Matrix<T>::Matrix(int rows, int cols, complex<T>) {
	_rows = rows;
	_cols = cols;
	_data = new complex<T>*[_rows];

	for (int i = 0; i < _rows; ++i) {
		_data[i] = new complex<T>[_cols];
	}

}
template<typename T>
Matrix<T>::Matrix(int rows, int cols, const complex<T>& lower, const complex<T>& upper) {
    _rows = rows;
    _cols = cols;
    _data = new complex<T>*[_rows];
    
    for (int i = 0; i < _rows; ++i) {
        _data[i] = new std::complex<T>[_cols];
    }

    srand(static_cast<unsigned>(time(nullptr)));

    for (int i = 0; i < _rows; ++i) {
        for (int j = 0; j < _cols; ++j) {
            T real_part = lower.real() + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX) / (upper.real() - lower.real()));
            T imag_part = lower.imag() + static_cast<T>(rand()) / (static_cast<T>(RAND_MAX) / (upper.imag() - lower.imag()));
            _data[i][j] = complex<T>(real_part, imag_part);
        }
    }
}
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other) : _rows(other._rows), _cols(other._cols) {
	_data = new T * [_rows];
	for (int i = 0; i < _rows; ++i) {
		_data[i] = new T[_cols];
		for (int j = 0; j < _cols; ++j) {
			_data[i][j] = other(i, j);
		}
	}
}
template<typename T>
Matrix<T>::Matrix(const Matrix<complex<T>>& other) : _rows(other._rows), _cols(other._cols) {
	_data = new T * [_rows];
	for (int i = 0; i < _rows; ++i) {
		_data[i] = new T[_cols];
		for (int j = 0; j < _cols; ++j) {
			_data[i][j] = other(i, j);
		}
	}
}
template<typename T>
Matrix<T>::~Matrix() {
	for (int i = 0; i < _rows; ++i) {
		delete[] _data[i];
	}
	delete[] _data;
}

template<typename T>
void Matrix<T>::set_rows(int rows) {
	this->_rows = rows;
}
template<typename T>
void Matrix<T>::set_cols(int cols) {
	this->_cols = cols;
}
template<typename T>
int Matrix<T>::get_rows() const {
	return _rows;
}
template<typename T>
int Matrix<T>::get_cols() const {
	return _cols;
}

template<typename T>
T& Matrix<T>::operator()(int row, int col) const {
	if (row > _rows || col > _cols) {
		throw invalid_argument("Invalid index of matrix");
	}
	return _data[row][col];
}
template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T>& other) const {
	return Matrix(*this) += other;
}
template<typename T>
Matrix<complex<T>> Matrix<T>::operator+(const Matrix<complex<T>>& other) const {
	return Matrix(*this) += other;
}
template<typename T>
Matrix<T> Matrix<T>::operator+=(const Matrix<T>& other) {
	if (_rows != other._rows || _cols != other._cols) {
		throw invalid_argument("Matrix dimensions must be the same for addition.");
	}

	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			_data[i][j] += other(i, j);
		}
	}
	return *this;
}
template<typename T>
Matrix<complex<T>> Matrix<T>::operator+=(const Matrix<complex<T>>& other) {
	if (_rows != other._rows || _cols != other._cols) {
		throw invalid_argument("Matrix dimensions must be the same for addition.");
	}

	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			_data[i][j].real() += other(i, j);
			_data[i][j].imag() += other(i, j);
		}
	}
	return *this;
}
template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T>& other) const {
	return Matrix(*this) -= other;
}
template<typename T>
Matrix<complex<T>> Matrix<T>::operator-(const Matrix<complex<T>>& other) const {
	return Matrix(*this) -= other;
}
template<typename T>
Matrix<T> Matrix<T>::operator-=(const Matrix<T>& other) {
	if (_rows != other._rows || _cols != other._cols) {
		throw invalid_argument("Matrix dimensions must be the same for subtraction.");
	}

	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			_data[i][j] -= other(i, j);
		}
	}
	return *this;
}
template<typename T>
Matrix<complex<T>> Matrix<T>::operator-=(const Matrix<complex<T>>& other) {
	if (_rows != other._rows || _cols != other._cols) {
		throw invalid_argument("Matrix dimensions must be the same for subtraction.");
	}

	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			_data[i][j] -= other(i, j);
		}
	}
	return *this;
}
template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T>& other) const {
	if (_cols != other._rows) {
		throw invalid_argument("Matrix dimensions must be the same for composition.");
	}
	int result_rows = _rows;
	int result_cols = other._cols;
	Matrix<T> result(result_rows, result_cols);

	for (int i = 0; i < result_rows; ++i) {
		for (int j = 0; j < result_cols; ++j) {
			for (int k = 0; k < _cols; ++k) {
				result(i, j) += _data[i][k] * other(k, j);
			}
		}
	}
	return result;

}
template<typename T>
Matrix<complex<T>> Matrix<T>::operator*(const Matrix<complex<T>>& other) const {
	if (_cols != other._rows) {
		throw invalid_argument("Matrix dimensions must be the same for composition.");
	}
	int result_rows = _rows;
	int result_cols = other._cols;
	Matrix<complex<T>> result(result_rows,result_cols, complex<T>);

	for (int i = 0; i < result_rows; ++i) {
		for (int j = 0; j < result_cols; ++j) {
			for (int k = 0; k < _cols; ++k) {
				result(i, j) += _data[i][k] * other(k, j);
			}
		}
	}
	return result;
}
template<typename T>
Matrix<T> Matrix<T>::operator*(const T& scalar) const {
	Matrix<T> result(_rows, _cols);
	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			result(i, j) = _data[i][j] * scalar;
		}
	}
	return result;

}
template<typename T>
Matrix<complex<T>> Matrix<T>::operator*(const complex<T>& scalar) const {
	Matrix<complex<T>> result(_rows, _cols);
	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			result(i, j) = _data[i][j] * scalar;
		}
	}
	return result;
}
template<typename T>
Matrix<T> Matrix<T>::operator/(const T& scalar) const {
	Matrix<T> result(_rows, _cols);
	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			result(i, j) = _data[i][j] / scalar;
		}
	}
	return result;

}
template<typename T>
Matrix<complex<T>> Matrix<T>::operator/(const complex<T>& scalar) const {
	Matrix<complex<T>> result(_rows, _cols);
	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			result(i, j) = _data[i][j] / scalar;
		}
	}
	return result;
}

template<typename T>
T Matrix<T>::trace() const {
	if (_rows != _cols) {
		throw invalid_argument("Matrix must be square for trace calculation.");
	}

	T result = T(0);
	for (int i = 0; i < _rows; ++i) {
		result += _data[i][i];
	}
	return result;
}
template<typename T>
bool Matrix<T>::operator==(const Matrix<T>& other) const {
	if (_rows != other._rows || _cols != other._cols)
		return false;
	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			if (_data[i][j] != other._data[i][j])
				return false;
		}
	}
	return true;
}
template<typename T>
bool Matrix<T>::operator==(const Matrix<complex<T>>& other) const {
	if (_rows != other._rows || _cols != other._cols)
		return false;
	for (int i = 0; i < _rows; ++i) {
		for (int j = 0; j < _cols; ++j) {
			if (_data[i][j] != other._data[i][j])
				return false;
		}
	}
	return true;
}
template<typename T>
bool Matrix<T>::operator!=(const Matrix<T>& other) const {
	return !(*this == other);
}
template<typename T>
bool Matrix<T>::operator!=(const Matrix<complex<T>>& other) const {
	return !(*this == other);
}
int main() {
	Matrix<int> matrix1(5, 5, 1, 8);
	Matrix<int> matrix2(5, 5, 1, 8);
	Matrix<float> matrix3(5, 5, 1.0, 5.0);
	Matrix<int> matrix(4, 3);
	Matrix<int> sum = matrix1 + matrix2;
	Matrix<int> substraction = matrix1 - matrix2;
	Matrix<int> multiplication = matrix1 * matrix2;
	Matrix<int> multiplication2 = matrix1 * 5;
	Matrix<int> division = matrix1/3;
	cout << "Zero matrix: \n" << matrix << endl;
	cout << "Matrix 1: \n" << matrix1 << endl;
	cout << "Matrix 2: \n" << matrix2 << endl;
	cout << "Matrix of float: \n" << matrix3 << endl;
	cout << "Sum of matrix: \n" << sum << endl;
	cout << "Substraction: \n" << substraction << endl;
	cout << "Multiplication: \n" << multiplication << endl;
	cout << "Multiplication by scalar: \n" << multiplication2 << endl;
	cout << "Division: \n" << division << endl;
	int trace = matrix1.trace();
	float trace2 = matrix3.trace();
	cout << "int trace: " << trace << endl;
	cout << "float trace: " << trace2 << endl;
	bool iseq = matrix1 == matrix2;
	bool isnteq = matrix1 != matrix2;
	cout << "is equal: " << iseq << endl;
	cout << "isn't equal: " << isnteq  << endl << endl;

	Matrix<complex<float>> complex1(2, 2, complex<float>(1.0f, 2.0f), complex<float>(5.0f, 4.0f));
	Matrix<complex<float>> complex2(2, 2, complex<float>(2.0f, 3.0f), complex<float>(4.0f, 6.0f));
	Matrix<complex<double>> double_complex(2, 2, complex<double>(1.0, 3.0), complex<double>(3.0, 6.0));
	Matrix<complex<float>> complex3 = complex1 + complex2;
	Matrix<complex<float>> complex4 = complex1 - complex2;
	Matrix<complex<float>> complex5 = complex1 * complex2;
	Matrix<complex<float>> complex6 = complex1 * 4;
	Matrix<complex<float>> complex7 = complex6 / 4;
	cout << "First complex matrix: \n" << complex1 << endl;
	cout << "Second complex matrix: \n" << complex2 << endl;
	cout << "Complex matrix with double: \n" << double_complex << endl;
	cout << "Sum of matrix: \n" << complex3 << endl;
	cout << "Substraction: \n" << complex4 << endl;
	cout << "Multiplication: \n" << complex5 << endl;
	cout << "Multiplication by scalar: \n" << complex6 << endl;
	cout << "Division: \n" << complex7 << endl;
	complex<float> trace_complex = complex1.trace();
	cout << "complex trace: " << trace_complex << endl;
	bool iseq2 = complex1 == complex2;
	bool isnteq2 = complex1 != complex2;
	cout << "is equal: " << iseq2 << endl;
	cout << "isn't equal: " << isnteq2 << endl << endl;
}