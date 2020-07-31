
var express=require("express");
var router=express.Router();
var Campground=require("../models/campground");
var middleware=require("../middleware");
//================
//CAMPGROUND ROUTES
//=================
function escapeRegex(text) {
    return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&");
};
//index-show all campgrounds
router.get("/campgrounds", function(req, res){
    if(req.query.search && req.xhr) {
        const regex = new RegExp(escapeRegex(req.query.search), 'gi');
        // Get all campgrounds from DB
        Campground.find({name: regex}, function(err, allCampgrounds){
           if(err){
              console.log(err);
           } else {
              res.status(200).json(allCampgrounds);
           }
        });
    } else {
        // Get all campgrounds from DB
        Campground.find({}, function(err, allCampgrounds){
           if(err){
               console.log(err);
           } else {
              if(req.xhr) {
                res.json(allCampgrounds);
              } else {
                res.render("campgrounds/index",{campgrounds: allCampgrounds, page: 'campgrounds'});
              }
           }
        });
    }
  });
 //create-add new campground
 router.post("/campgrounds",middleware.isLoggedIn,function(req,res)
 {
  var name=req.body.name;
  var price=req.body.price;
  var image=req.body.image;
  var desc=req.body.description;

  var author={
      id:req.user._id,
      username:req.user.username
  }
  var newCampground={name:name,price:price,image:image,description:desc,author:author}
 Campground.create(newCampground,function(err,newlyCreated){
     if(err){
         console.log(error);
     }else{
         res.redirect("/campgrounds"); 
     }
 });
     
 });
 
 //new-show form to create new campground
 router.get("/campgrounds/new",middleware.isLoggedIn,function(req,res){
     res.render("campgrounds/new");
 });
 
 //show-shows more info about each campground
 router.get("/campgrounds/:id",function(req,res){
     Campground.findById(req.params.id).populate("comments").exec(function(err,foundCampground){
     res.render("campgrounds/show",{campground:foundCampground});
    });
     });

 //Edit Campground route
 router.get("/campgrounds/:id/edit",middleware.checkCampgroundOwnership,function(req,res){  
     Campground.findById(req.params.id,function(err,foundCampground){
         if(err){
             res.redirect("/campgrounds");
         } else{
            res.render("campgrounds/edit",{campground:foundCampground});
         }
     });
 });


 //Update Campground route
 
router.put("/campgrounds/:id",middleware.checkCampgroundOwnership,function(req,res){
//find and update the correct campground
Campground.findByIdAndUpdate(req.params.id,req.body.campground,function(err,updatedCampground){
    if(err){
        res.redirect("/campgrounds");
   }else{
        res.redirect("/campgrounds/"+ req.params.id);
    }
});
});

//Destroy campground route
router.delete("/campgrounds/:id",middleware.checkCampgroundOwnership,function(req,res){
    Campground.findByIdAndRemove(req.params.id,function(err){
        if(err){
            res.redirect("/campgrounds");
        } else{
            res.redirect("/campgrounds");
        }
    });

});


 module.exports=router;