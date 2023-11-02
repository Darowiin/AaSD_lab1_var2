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
	complex<T> complex_trace() const;

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
		_data[i] = new T[_cols];
		for (int j = 0; j < _cols; ++j) {
			_data[i][j] = 0;
		}
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
T& Matrix<T>::operator()(int row, int col) const{
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
		throw std::invalid_argument("Matrix dimensions must be the same for addition.");
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
		throw std::invalid_argument("Matrix dimensions must be the same for addition.");
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
		throw std::invalid_argument("Matrix dimensions must be the same for subtraction.");
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
		throw std::invalid_argument("Matrix dimensions must be the same for subtraction.");
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
	int result_rows = _rows;
	int result_cols = other._cols;
	Matrix<complex<T>> result(result_rows, result_cols, T(0));

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
complex<T> Matrix<T>::complex_trace() const {
	if (_rows != _cols) {
		throw invalid_argument("Matrix must be square for trace calculation.");
	}

	complex<T> result = complex<T>(0);
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
	Matrix<int> multiplication = matrix1 * matrix2;
	Matrix<int> multiplication2 = matrix1 * 5;
	Matrix<int> division = matrix1/3;
	cout << "Zero matrix: \n" << matrix << "\n";
	cout << "Matrix 1: \n" << matrix1 << "\n";
	cout << "Matrix 2: \n" << matrix2 << "\n";
	cout << "Matrix of float: \n" << matrix3 << "\n";
	cout << "Sum of matrix: \n" << sum << "\n";
	cout << "Multiplication: \n" << multiplication << "\n";
	cout << "Multiplication by scalar: \n" << multiplication2 << "\n";
	cout << "Division: \n" << division << "\n";
	int trace = matrix1.trace();
	float trace2 = matrix3.trace();
	cout << "int trace: " << trace << "\n";
	cout << "float trace: " << trace2 << "\n";
	bool iseq = matrix1 == matrix2;
	bool isnteq = matrix1 != matrix2;
	cout << "is equal: " << iseq << "\n";
	cout << "isn't equal: " << isnteq << "\n";
}