var express=require("express"),
    bodyParser=require("body-parser"),
    app=express(),
    mongoose=require("mongoose"),
    flash=require("connect-flash"),
    passport=require("passport"),
    LocalStratergy=require("passport-local"),
    methodOverride=require("method-override"),
    Campground=require("./models/campground"),
    Comment=require("./models/comment"),
    User=require("./models/user"),
    seedDB=require("./seeds");

app.use(methodOverride('_method'));
app.use(flash());

var commentRoutes=require("./routes/comments"),
    campgroundRoutes=require("./routes/campgrounds"),
    Auth=require("./routes/attendance"),
    authRoutes=require("./routes/index");


//Seeding the database
//seedDB();

//PASSPORT CONFIGURATION

app.use(require("express-session")({
     secret:"Once again Tejas is the best",
     resave:false,
     saveUninitialized:false
}));


app.use(passport.initialize());
app.use(passport.session());
passport.use(new LocalStratergy(User.authenticate()));
passport.serializeUser(User.serializeUser());
passport.deserializeUser(User.deserializeUser());

app.use(function(req,res,next){
    res.locals.currentUser=req.user;
    res.locals.error=req.flash("error");
    res.locals.success=req.flash("success");
    next();
});

app.use(bodyParser.urlencoded({extended: true}));
app.set("view engine","ejs");
app.use(express.static(__dirname+"/public"));
mongoose.set('useFindAndModify',false);

let MongodbURL="mongodb+srv://tejas:9811184482@yelpcamp.zqhq4.mongodb.net/test?retryWrites=true&w=majority"
mongoose.connect(MongodbURL || "mongodb://localhost:27017/yelp_camp",
{useNewUrlParser:true,useUnifiedTopology: true});

app.use(commentRoutes);
app.use(campgroundRoutes);
app.use(Auth);
app.use(authRoutes);
app.listen(process.env.PORT||3000,process.env.IP||"0.0.0.0",function(){
    console.log("server started");
});