using BrnoAiDaysWorkshop;
using Microsoft.Extensions.ML;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();
builder.Services.AddPredictionEnginePool<InputImageData, PredictionModel>().FromFile($"{builder.Environment.ContentRootPath}..\\BrnoAiDaysWorkshop\\bin\\Debug\\net6.0\\model\\trained_model");

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

app.UseHttpsRedirection();
app.MapPost(
    "evaluate/{filePath:required}",
    (string filePath, PredictionEnginePool<InputImageData, PredictionModel> enginePool) =>
    {
        var predictionEngine = enginePool.GetPredictionEngine();
        var predict = predictionEngine.Predict(new InputImageData() { ImagePath = filePath.Trim('"') });
        return predict.PredictedLabel;
    });


app.Run();