#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include "NN.h"

using namespace std;

int main() {
	vector<NN::vector> X,y;
	//for(int a = 0; a < 2; a++) for(int b = 0; b < 2; b++) X.push_back(NN::vector({(double)a,(double)b})), y.push_back(NN::vector({(double)(a^b)}));
	for(double a = -1; a < 1; a+=0.1) {
		for(double b = -1; b < 1; b += 0.1) {
			X.push_back(NN::vector({a, b}));
			y.push_back(NN::vector({sin(2*a*b)}));
		}
	}
	NN::network net({
		NN::layer("relu", 3, 2),
		NN::layer("relu", 3, 3),
		NN::layer("relu", 1, 3),
	});
	NN::automatic_fit(net, X, y, 150, 0.02);
	for(int i = 0; i < X.size(); i++) {
		cout << X[i][0] << " " << X[i][1] << " " << y[i][0] << " : " << net(X[i]).back()[0] << "\n";
	}
	cout << net << "\n";
}