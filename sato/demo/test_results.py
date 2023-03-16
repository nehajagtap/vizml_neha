import unittest
import test_main_helper

class TestClass(unittest.TestCase):
    
    file_path = "/media/saadnajib/56ce7057-5d73-4456-a6b0-7a325a03ba0f/saad/checking_datatype/election.csv"
    
    def test_city(self):
       
        self.assertEqual(test_main_helper.test_city(TestClass.file_path,"STRCITY"),True)
        self.assertEqual(test_main_helper.test_city(TestClass.file_path,"STREETADDR"),False)

    def test_address(self):

        self.assertEqual(test_main_helper.test_address(TestClass.file_path,"STREETADDR"),True)
        self.assertEqual(test_main_helper.test_address(TestClass.file_path,"STRCITY"),False)

    def test_date_and_time(self):

        self.assertEqual(test_main_helper.test_date_and_time(TestClass.file_path,"CallDate"),True)   
        self.assertEqual(test_main_helper.test_date_and_time(TestClass.file_path,"STRCITY"),False)       

    def test_name(self):

        self.assertEqual(test_main_helper.test_name(TestClass.file_path,"CallerSurname"),True)
        self.assertEqual(test_main_helper.test_name(TestClass.file_path,"CallDuration"),False)    

    def test_type(self):
        
        self.assertEqual(test_main_helper.test_type(TestClass.file_path,"type_store"),True)
  
    def test_location(self):
        
        self.assertEqual(test_main_helper.test_location(TestClass.file_path,"ZIPCODE"),True)

    def test_number(self):
        
        self.assertEqual(test_main_helper.test_number(TestClass.file_path,"CalledNumber"),True)

    def test_year(self):
        
        self.assertEqual(test_main_helper.test_year(TestClass.file_path,"YEAR"),True)

    def test_day(self):
        
        self.assertEqual(test_main_helper.test_day(TestClass.file_path,"DAY"),True)

    def test_month(self):
        
        self.assertEqual(test_main_helper.test_month(TestClass.file_path,"MONTH"),True)

    def test_cordinates(self):
        
        self.assertEqual(test_main_helper.test_cordinates(TestClass.file_path,"MONTH"),False)
        self.assertEqual(test_main_helper.test_cordinates(TestClass.file_path,"CalledLongitude"),True)
        self.assertEqual(test_main_helper.test_cordinates(TestClass.file_path,"CalledLatitude"),True)
   
    def test_duration(self):

        self.assertEqual(test_main_helper.test_duration(TestClass.file_path,"CallTime"),False)
        self.assertEqual(test_main_helper.test_duration(TestClass.file_path,"CallDuration"),True)

    def test_gender(self):

        self.assertEqual(test_main_helper.test_gender(TestClass.file_path,"Gender"),True)

    def test_status(self):

        self.assertEqual(test_main_helper.test_status(TestClass.file_path,"Status"),True)

    def test_crime(self):

        self.assertEqual(test_main_helper.test_crime(TestClass.file_path,"Crime"),True)

if __name__ == '__main__':
    unittest.main()
