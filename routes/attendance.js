var express=require("express");
var router=express.Router();

router.get("/attendance",function(req,res){
    res.render("attendance");
});


router.get("/attendance/add",function(req,res){
    res.render("addAttendance")
});

router.get("/attendance/add/result",function(req,res){
      res.render("attendanceResult");
})






module.exports=router;