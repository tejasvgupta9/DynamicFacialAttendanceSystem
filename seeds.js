var mongoose=require("mongoose");
var Campground=require("./models/campground");
var Comment=require("./models/comment");


var data=[
    {
        name:"Clouds Rest",
        image:"https://images.pexels.com/photos/699558/pexels-photo-699558.jpeg?auto=compress&cs=tinysrgb&h=350",
        description:"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras mollis nisi id lorem faucibus convallis. Ut ut dolor ut neque placerat bibendum quis sit amet elit. Fusce egestas dui in venenatis ultricies."
    },
    
    {
        name:"Desset Mesa",
        image:"https://images.pexels.com/photos/2422265/pexels-photo-2422265.jpeg?auto=compress&cs=tinysrgb&h=350",
        description:"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras mollis nisi id lorem faucibus convallis. Ut ut dolor ut neque placerat bibendum quis sit amet elit. Fusce egestas dui in venenatis ultricies."
    },
    
    {
             name:"Canyon Floor",
        image:"https://images.pexels.com/photos/776117/pexels-photo-776117.jpeg?auto=compress&cs=tinysrgb&h=350",
        description:"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras mollis nisi id lorem faucibus convallis. Ut ut dolor ut neque placerat bibendum quis sit amet elit. Fusce egestas dui in venenatis ultricies."
    },
    
];


function seedDB(){
Campground.remove({},function(err){
    if(err){
        console.log(err);
    }
   console.log("removed campgrounds");
     //ADD a few Campgrounds
     data.forEach(function(seed){
       Campground.create(seed,function(err,campground){
           if(err){
                console.log(error);
            } else {
                console.log("added a campground");
               Comment.create(
                   {
                        desc:"This place is great,I wish there was Internet",
                        author:"Homer"
                },function(err,comment){
                      campground.comments.push(comment);
                         if(err){
                                console.log(err);
                           } else {
                               campground.comments.push(comment);
                                campground.save();
                                console.log("Created New Comment");
                                }
                
                 });
                   
            }
           });
        });
    });
}

module.exports=seedDB;