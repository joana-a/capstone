<?php

include("../controllers/user_controller.php");


// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Retrieve form data
    $username = $_POST["username"];
    $email = $_POST["email"];
    $password = $_POST["password"];
    $gender = $_POST["gender"]; 
    $contact = $_POST["contact"];
 

    $registrationResult = registerUser_ctr($username, $email, $password, $gender, $contact);


    if ($registrationResult !== false) {
        // Registration successful
        header("Location: ../login/login.php");
    } else {
        // Registration failed
        $error_message = "Registration failed. ";
        echo $error_message . "Handle errors or redirect as needed.";
    }
}










