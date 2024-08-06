import express from "express";
import ntc from "ntc";
import cors from "cors"
import { pipeline } from '@xenova/transformers';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express()
app.use(cors())

app.use(express.static(__dirname));

app.get('/sentiment', async function (req, res) {
    
    let classifier = await pipeline('sentiment-analysis');
    let result = await classifier('I love transformers!');
    console.log(result)
    var n_match = ntc.name('#6195ED')
    console.log(n_match)
    res.send('changed this quickly')
})

app.get('/index', async function (req, res) {
    
   res.sendFile(__dirname + '/index.html')
})

app.listen(3000)