library(shiny)
library(DT)
library(shinythemes)



#vector defines 
outputDir <- "responses"


#function that has the instructions on how to save to a csv file
saveData <- function(data) {
  data <- t(data)
  # Create a unique file name
  fileName <- sprintf("%s_%s.csv", as.integer(Sys.time()), digest::digest(data))
  # Write the file to the local system
  write.csv(
    x = data,
    file = file.path(outputDir, fileName), 
    row.names = FALSE, quote = TRUE
  )
}


#instructions on how to load the data
loadData <- function() {
  # Read all the files into a list
  files <- list.files(outputDir, full.names = TRUE)
  data <- lapply(files, read.csv, stringsAsFactors = FALSE) 
  # Concatenate all data together into one data.frame
  data <- do.call(rbind, data)
  data
}

fields2 <- c("close_or_far") 
fields3 <- c("light_or_dark")

fields4 <- c("behind_glass")

fields5 <- c("has_truck") 
fields6 <- c("has_snow") 
fields7 <- c("image_blurry") 
fields8 <- c("image_perpendicular") 



fields9 <- c("snow_on_rock") 

fields10 <- c("user_id_input") 

fields11 <- c("timestamp")

fields12 <- c("mask_good_or_bad") 



fields1 <- c("file_name")

#Directory where the images are saved 

img_source <- "./original/"

img_source_train <- "./superimposed_train_labelme/"

img_source_predict <- "./train_predict/"




#Reads the image files for original
imgs <- list.files(img_source,full.names = TRUE)

#removes the filepath for the csv file
imgs2 <- gsub(img_source,"",imgs)


#Reads the image files for labelled images
imgs_train <- list.files(img_source_train,full.names = TRUE)

#removes the filepath for the csv file
#imgs_train2 <- gsub(img_source_train,"",imgs_train)

#Reads the image files for predictions
imgs_predict <- list.files(img_source_predict,full.names = TRUE)

#removes the filepath for the csv file
#imgs_predict2<- gsub(img_source_predict,"",imgs_predict)


ui <- fluidPage(theme = shinytheme("slate"),

  #Download all responses
  
  
  titlePanel("image Quality Assurance (iQA)"),
  
  sidebarLayout(
    sidebarPanel(
      actionButton("previous", "Previous"),
      actionButton("next", "Next"),
      
      
      #saving the results button
      actionButton("save", "Save"),
      
      
      #Enter the user Id button
      textInput("user_id_input", h3("User ID"), 
                value = "Enter User ID..."),
      
      #button for reading in the file, has a header checkbox
      fileInput('file1', 'Choose CSV File for image list',
                accept=c('text/csv', 'text/comma-separated-values,text/plain', '.csv')),
      checkboxInput('header', 'Header in csv file present?', TRUE),
      
      
      #qualitative analysys buttons(lots of repetition)
      radioButtons("mask_good_or_bad", HTML("<h4>Does the prediction overlap the label?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Poor Overlap</b></p>"),
                     HTML("<p style='color:green;'><b>YES There is good overlap</b></p>")
                   ),
                   choiceValues = list(
                     "Poor_Overlap", "perfect"
                   )),
      
      
      radioButtons("close_or_far", HTML("<h4>Is picture too close or far?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Too Far</b></p>"),
                     HTML("<p style='color:green;'><b>Perfect</b></p>"),
                     HTML("<p style='color:red;'><b>Too Close</b></p>")
                   ),
                   choiceValues = list(
                     "far", "perfect", "close"
                   )),
      
      radioButtons("light_or_dark",  HTML("<h4>Is picture too light or dark?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Too Light</b></p>"),
                     HTML("<p style='color:green;'><b>Perfect</b></p>"),
                     HTML("<p style='color:red;'><b>Too Dark</b></p>")
                   ),
                   choiceValues = list(
                     "light", "perfect", "dark"
                   )),
      
      radioButtons("behind_glass", HTML("<h4>Is picture behind glass?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Behind Glass</b></p>"),
                     HTML("<p style='color:green;'><b>No Glass</b></p>")
                   ),
                   choiceValues = list(
                     "behind_glass", "perfect"
                   )),
      
      
      radioButtons("has_truck", HTML("<h4>Does picture have truck/machinery ?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Has Truck / Machinery</b></p>"),
                     HTML("<p style='color:green;'><b>No Truck</b></p>")
                   ),
                   choiceValues = list(
                     "has_truck_machinery", "perfect"
                   )),
      
      radioButtons("has_snow", HTML("<h4>Does picture have snowflakes?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Has snowflakes</b></p>"),
                     HTML("<p style='color:green;'><b>No snowflakes</b></p>")
                   ),
                   choiceValues = list(
                     "has_snowflakes", "perfect"
                   )),
      
      radioButtons("image_blurry", HTML("<h4>Is picture blurry?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Blurry</b></p>"),
                     HTML("<p style='color:green;'><b>Sharp</b></p>")
                   ),
                   choiceValues = list(
                     "Blurry", "perfect"
                   )),
      radioButtons("image_perpendicular", HTML("<h4>Is rock face angled?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Angled</b></p>"),
                     HTML("<p style='color:green;'><b>Flat</b></p>")
                   ),
                   choiceValues = list(
                     "Angled", "perfect"
                   )),
      radioButtons("snow_on_rock", HTML("<h4>Is there snow on rock face?</h4>"),
                   choiceNames = list(
                     HTML("<p style='color:red;'><b>Yes</b></p>"),
                     HTML("<p style='color:green;'><b>No Snow</b></p>")
                   ),
                   choiceValues = list(
                     "Snow_present", "perfect"
                   )),
  
    
      downloadButton("summary", "Download Responses"),

      actionButton("delete", "Delete Contents")
      
    ),
    
    mainPanel(
      #I cant seem to render the logo file
      #img(src='iron_ore_logo.jpg', align = "right"),
      h3("Original Image:", align = "center"),
      imageOutput("image"),
      h3("Image with Mask via Labelme:", align = "center"),
      imageOutput("image2"),
      h3("Prediction using the model:", align = "center"),
      imageOutput("image3"),
      h2("How to use this app:", align = "center"),
      HTML("<p>The program will naturally go to the image repository folder</p>"),
      HTML("<p>If you want to look at specific images, you can upload a csv file with the filenames</p>"),
      HTML("<p>Select the definitions as shown on the left hand side.</p>"),
      HTML("<p>Once you are happy wih your choice click <b>Save</b> to save the data for that picture.</p>"),
      HTML("<p>Then click <b>Next</b> to go to the next picture.</p>"),
      HTML("<p>Once you have gone through all the images, click the <b>Download Responses</b> button to save the csv with all of the results.</p>"),
      HTML("<p>To delete all of the past analyses click the <b>Delete Contents</b> button to empty the response folder</p>")
    )
  )
)

server <- function(input, output, session) {
  
  
  
  #Will read a csv with files and will cycle through the list. If the image is not there, there will be nothing to display
  #It is done 3 rtimes for each of the timage
  
  
  getdata <- reactive({
    #inFile <- as.character(input$file1)
    inFile <- input$file1
    
    if (is.null(inFile))
      inFile <- imgs
    #return(NULL)
    else
      #fucking complicated, but I need to document it. 3 days for 5 lines of code 
      paste0( img_source, as.vector(unlist(read.csv(inFile$datapath, header = input$header))))
    #Basically it is 3 functions wrapped into 1
    #1
    #The character vector does not know the directory path of the file
    #   inFile <- paste0( img_source, inFile)
    #2
    #the reactive function needs to read the filesnames as a character vector and now it is 
    #turned into a character vector
    #   inFile <- as.vector(unlist(inFile))
    #3
    #read csv turns the reactive input into a list
    #   inFile <-read.csv(inFile$datapath, header = FALSE)
  })
  
  getdata_train <- reactive({
    inFile_train <- input$file1
    
    if (is.null(inFile_train))
      inFile_train <- imgs_train
    #return(NULL)
    else
      #fucking complicated, but I need to document it. 3 days for 5 lines of code 
      paste0( img_source_train, as.vector(unlist(read.csv(inFile_train$datapath, header = input$header))))
    #Basically it is 3 functions wrapped into 1
    #1
    #The character vector does not know the directory path of the file
    #   inFile <- paste0( img_source, inFile)
    #2
    #the reactive function needs to read the filesnames as a character vector and now it is 
    #turned into a character vector
    #   inFile <- as.vector(unlist(inFile))
    #3
    #read csv turns the reactive input into a list
    #   inFile <-read.csv(inFile$datapath, header = FALSE)
  })
  
  
  
  getdata_predict <- reactive({
    #inFile <- as.character(input$file1)
    inFile_predict <- input$file1
    
    if (is.null(inFile_predict))
      inFile_train <- imgs_predict
    #return(NULL)
    else
      #fucking complicated, but I need to document it. 3 days for 5 lines of code 
      paste0( img_source_predict, as.vector(unlist(read.csv(inFile_predict$datapath, header = input$header))))
    #Basically it is 3 functions wrapped into 1
    #1
    #The character vector does not know the directory path of the file
    #   inFile <- paste0( img_source, inFile)
    #2
    #the reactive function needs to read the filesnames as a character vector and now it is 
    #turned into a character vector
    #   inFile <- as.vector(unlist(inFile))
    #3
    #read csv turns the reactive input into a list
    #   inFile <-read.csv(inFile$datapath, header = FALSE)
  })

  
  index <- reactiveVal(1)
  # When the next button is clicked, it shows the previous image
  observeEvent(input[["previous"]], {
    index(max(index()-1, 1))
  })
  # When the next button is clicked, it shows the next image
  observeEvent(input[["next"]], {
    index(min(index()+1, length(getdata())))
  })
  # When the save button is clicked, it also saves the form data
  observeEvent(input[["save"]], {
    saveData(formData())
  })
  
  # When the delete responses button is clicked,it deletes the data in the response folder
  observeEvent(input[["delete"]], {
    do.call(file.remove, list(list.files(outputDir, full.names = TRUE))) 
  })
  
  output$summary <- downloadHandler(
    filename = function() { 
      sprintf("mimic-google-form_%s.csv", as.integer(Sys.time()))
    },
    content = function(file) {
      write.csv(loadData(), file, row.names = FALSE)
    }
  )
  
  # Whenever a field is filled, aggregate all form data
  formData <- reactive({
    
   
   #bad_image <- sapply(fields1, function(x) input[[x]])
   close_far <- sapply(fields2, function(x) input[[x]])
   light_dark <- sapply(fields3, function(x) input[[x]])
   behind_glass <- sapply(fields4, function(x) input[[x]])
   has_truck <- sapply(fields5, function(x) input[[x]])
   has_snow <- sapply(fields6, function(x) input[[x]])
   image_blurry <- sapply(fields7, function(x) input[[x]])
   image_perpendicular <- sapply(fields8, function(x) input[[x]])
   snow_on_rock <- sapply(fields9, function(x) input[[x]])
   date_stamp <-  sapply(fields11, function(x) format(Sys.time(), format="%a %b %d %X %Y"))
   mask_good_or_bad <- sapply(fields12, function(x) input[[x]])
   
   
   #No need to use toString
   
   user_id_input <- sapply(fields10, function(x) input[[x]])
   
  
   
   #Reads the filename and stores it in a list  
   files <- sapply(fields1, function(x) imgs2[index()])
    
    #combining all of the lists into a single dataframe
   data <- rbind(files, close_far, light_dark, behind_glass, has_truck, has_snow, image_blurry, image_perpendicular, snow_on_rock,  mask_good_or_bad, user_id_input, date_stamp)

  })
  
  
  
  
  
  #How to render the image (original)
  output$image <- renderImage({
    x <- getdata()[index()] 
    list(src = x, height="100%", width="100%", align="right",alt = "alternate text")
  }, deleteFile = FALSE)
  
  #How to render the image (labelMe)
  output$image2 <- renderImage({
    x <- getdata_train()[index()] 
    list(src = x, height="100%", width="100%", align="right",alt = "alternate text")
  }, deleteFile = FALSE)
  #How to render the image (predicted images)
  output$image3 <- renderImage({
    x <- getdata_predict()[index()] 
    list(src = x, height="100%", width="100%", align="right",alt = "alternate text")
  }, deleteFile = FALSE)
  
}

# Run the application 
shinyApp(ui = ui, server = server)