## Problem Statement   

Predict the list of two other items/products which are frequently bought together whenever an item/product is added to the cart.<br>
  
## DataSet  

The dataset contains transactions on an e-commerce website between the period Feb 2018 to Feb 2019 from customers across different countries.

- UserId - Unique identifier of a user.  TransactionId - Unique identifier of a transaction. If the same TransactionId is present in multiple rows, then all those products are bought together in the same transaction.<br> 
- TransactionTime - Time at which the transaction is performed<br>  
- ItemCode - Unique identifier of the product purchased<br>   
- ItemDescription - Simple description of the product purchased<br>   
- NumberOfItemsPurchased - Quantity of the product purchased in the transaction<br>   
- CostPerItem - Price per each unit of the product<br>   
- Country - Country from which the purchase is made.<br>

## Summary


I analyzed the data first and differentiated the data according to the Country_basis, Sales-basis and number_of_sales basis.After analyzing and visualizing the whole data I decided to go through frequency-Matrix approach.I created my own Machine Learning Model from scrap using the the TransactionId and ItemCode.<br>
Frequency_Matrix is an NXN matrix. ( where N is the number of unique items)
First,
I created a Dictionary which Assigns each Item Code to a simpler code between 0 to 3406 [ Total Number of Unique Items is 3407]
Then Using my Fit Function I Created the Matrix.
Matrix stores the frequency of each item bought together.
i.e. Matrix[3][4] = 11
means Items with Simpler code 3 and 4 are bought together 11 times.
Second,
I added to new features month_year and Total Cost which helped in better understanding of the data. Description of each Function is provided in the notebook also.<br>

My Model used 3 functions:
- Fit (Takes two Panda data Frame columns as input)<br>
- Predict (Takes one item at a time)<br>
- Name<br>
The overall Model works on the Principle of common ness between two products which is analyzed using the vast dataset. Each time a Transaction Contains 2 or more items it changes the values in the matrix and that matrix in the end is used to determine or recommend the next two products for that item


