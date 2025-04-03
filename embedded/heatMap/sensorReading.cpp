// file for reading in sensor data
#include <iostream>
#include <chrono>
#include <thread>


//mock sensor data
int main(){
    mockData();
}

// In order for the python file to be able to call this function correctly we MUST
// create an executable file using a compiler (g++, gcc) and we will create a subprocess
// of that compiled c++ file in out python file.

// Compile in terminal using:
// g++ fileName.cpp -o compiledFileName





//Here is a function that creates mock data to parse over to the py file for interpratation:
int mockData(){
    while(true){
        //created simulated data (this is one row of a matrix)
        
        std::cout << "1.0,2.0,3.0" << std::endl;

        //flush data before any other data is created to ensure current data is sent before being overwritenn
        std::flush(std::cout);

        //let the program sleep until sending the next mock data
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}