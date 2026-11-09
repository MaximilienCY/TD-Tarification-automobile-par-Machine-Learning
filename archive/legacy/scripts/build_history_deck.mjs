// Editable presentation. Uses the Codex bundled Artifact Tool runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const CANVA_COMPATIBLE=process.argv.includes('--canva-compatible');
const BUILD=path.join(ROOT,'tmp/presentation_histoire',...(CANVA_COMPATIBLE?['canva_compatible']:[]));
const OUT=path.join(ROOT,'deliverables/presentation_histoire');
const RUNTIME='/home/cytech/.cache/codex-runtimes/codex-primary-runtime/dependencies';
process.env.RUNTIME_NODE_MODULES=path.join(RUNTIME,'node/node_modules');
const SKILL='/home/cytech/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(path.join(BUILD,'previews'),{recursive:true});
await fs.mkdir(OUT,{recursive:true});
const req=createRequire(path.join(RUNTIME,'node/node_modules','_history.cjs'));
const {createCanvas,GlobalFonts}=req('@napi-rs/canvas');
GlobalFonts.registerFromPath('/usr/share/fonts/truetype/lato/Lato-Regular.ttf','Lato');
GlobalFonts.registerFromPath('/usr/share/fonts/truetype/lato/Lato-Bold.ttf','Lato');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(RUNTIME,'node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const model=JSON.parse(await fs.readFile(path.join(OUT,'contenu_presentation.json'),'utf8'));
const ctx=createCanvas(20,20).getContext('2d');
const C={bg:'#FCFCFA',ink:'#172636',blue:'#2165A6',green:'#398454',muted:'#526476',rule:'#CFD7DB',pale:'#EAF1F6',sev:'#A95731'};
const P=Presentation.create({slideSize:{width:1280,height:720}});
const textEvidence=[];
let counter=0;
let currentNumber=0;
function wrap(value,w,size=24,bold=false){
  ctx.font=`${bold?'bold ':''}${size}px Lato`;
  const result=[];
  const safe=String(value).replace(/(\d) (?=\d{3}\b)/g,'$1\u00a0').replace(/(\d) (?=[%€])/g,'$1\u00a0');
  for(const para of safe.split('\n')){
    let line='';
    for(const word of para.split(' ')){
      if(!word){continue;}
      const next=line?`${line} ${word}`:word;
      if(line&&ctx.measureText(next).width>w){result.push(line);line=word;}else line=next;
    }
    result.push(line);
  }
  return result.join('\n');
}
function txt(slide,value,x,y,w,size=24,color=C.ink,bold=false,options={}){
  const s=options.noWrap?String(value):wrap(value,w,size,bold);
  const lines=s.split('\n').length;
  const h=options.height??(lines*size*1.25+8);
  if(y+h>716||x<0||x+w>1281)throw Error(`Text outside slide: ${value}`);
  const o=slide.shapes.add({geometry:'textbox',name:`text-${++counter}`,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  o.text=s;
  o.text.style={typeface:'Lato',fontSize:size,bold,color,wrap:'none',autoFit:'none',insets:0,
                verticalAlignment:'top',alignment:options.align??'left',lineSpacing:1.15};
  textEvidence.push({slide:currentNumber,text:s,position:{x,y,w,h},size});
  return {shape:o,height:h};
}
function rect(slide,x,y,w,h,fill,stroke='none',strokeWidth=0){
 return slide.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:stroke,width:strokeWidth}});
}
function line(slide,x,y,w,color=C.rule,thickness=1){return rect(slide,x,y,w,thickness,color);}
function connect(slide,a,b,color=C.blue,fromSide='right',toSide='left'){
 return slide.shapes.connect(a,b,{kind:'elbow',fromSide,toSide,line:{style:'solid',fill:color,width:2},tail:{type:'arrow',width:'med',length:'med'}});
}
function base(data){
 const s=P.slides.add();s.background.fill=C.bg;
 txt(s,data.stage.toUpperCase()+(data.appendix?' · QUESTIONS ET DÉTAILS':''),64,27,1000,16,C.blue,true);
 txt(s,`${String(data.number).padStart(2,'0')} / ${model.slides.length}`,1110,27,106,16,C.muted,false,{align:'right'});
 const title=txt(s,data.title,64,65,1152,42,C.ink,true);
 let leadTop=Math.max(134,65+title.height+10);
 if(data.lead)txt(s,data.lead,64,leadTop,1152,23,C.muted);
 line(s,64,225,1152,C.rule);
 if(data.takeaway){line(s,64,637,52,C.green,3);txt(s,data.takeaway,64,650,1152,23,C.ink,true);}
 s.speakerNotes.text=`${data.appendix?'ANNEXE POUR LES QUESTIONS':`REPÈRE DE TEMPS : ${data.seconds} SECONDES`}\n\n${data.notes}\n\nSOURCES\n`+
 data.refs.map(k=>{const r=model.sources[k];return `${r.label}\n${r.url??r.path}`;}).join('\n\n');
 return s;
}
function pairs(slide,items,x,y,w,maxH=350,size=25){
 let cy=y;
 for(const [label,body] of items){
   const h1=txt(slide,label,x,cy,w,21,C.blue,true).height;
   cy+=h1+3;
   const h2=txt(slide,body,x,cy,w,size).height;
   cy+=h2+19;
 }
 if(cy-y>maxH+30)console.log('DENSE COLUMN',items[0]?.[0],cy-y,maxH);
}
function table(slide,data,rows,x=64,y=250,w=1152,maxH=355){
 const count=rows[0].length;
 let weights=count===2?[.48,.52]:count===3?[.28,.39,.33]:count===4?[.36,.22,.21,.21]:[.43,.14,.14,.15,.14];
 if(data.number===12)weights=[.47,.13,.12,.28];
 if(data.number===29)weights=[.43,.18,.18,.21];
 if(data.number===19)weights=[.32,.35,.17,.16];
 if(data.number===25)weights=[.38,.2,.25,.17];
 if(count===5)weights=[.44,.14,.14,.15,.13];
 const widths=weights.map(a=>a*w);
 const font=data.compact?20:23;
 let gap=data.compact?6:10;
 const wrapped=rows.map((row,i)=>row.map((v,j)=>wrap(v,widths[j]-gap*2,font,i===0)));
 let heights=wrapped.map(row=>Math.max(...row.map(v=>v.split('\n').length))*font*1.12+gap*2);
 let total=heights.reduce((a,b)=>a+b,0);
 if(total>maxH){
   gap=Math.max(2,(maxH-(total-rows.length*2*gap))/(2*rows.length));
   heights=wrapped.map(row=>Math.max(...row.map(v=>v.split('\n').length))*font*1.12+gap*2);
   total=heights.reduce((a,b)=>a+b,0);
 }
 // Canva's native table importer changes padding and minimum row heights.
 // Keep native tables in the final PowerPoint; the private Canva import copy
 // uses editable text and rules, with the same values and row measurements.
 if(CANVA_COMPATIBLE){
  const totalHeight=heights.reduce((a,b)=>a+b,0);
  rect(slide,x,y,w,totalHeight,C.bg,C.rule,.65);
  rect(slide,x,y,w,heights[0],C.pale);
  let cy=y;
  for(let i=0;i<rows.length;i++){
   let cx=x;
   for(let j=0;j<count;j++){
    txt(slide,wrapped[i][j],cx+8,cy+2,widths[j]-16,font,i===0?C.blue:C.ink,i===0,{noWrap:true});
    if(i===0&&j>0)rect(slide,cx,y,.65,totalHeight,C.rule);
    cx+=widths[j];
   }
   if(i>0)line(slide,x,cy,w,C.rule,.65);
   cy+=heights[i];
  }
  return;
 }
 if(total>maxH){
   // Dense appendix matrices retain all rows, with a compact but editable layout.
   if(count===5){heights=heights.map(h=>h*maxH/total);}
   else console.log('TABLE OVERFLOW CANDIDATE',data.number,total,maxH);
 }
 const t=slide.tables.add({rows:rows.length,columns:count,left:x,top:y,width:w,height:heights.reduce((a,b)=>a+b,0),columnWidths:widths,values:wrapped});
 t.borders.assign({fill:C.rule,width:.65,style:'solid'});
 t.styleOptions={headerRow:true,bandedRows:false};
 t.cells.block({row:0,column:0,rowCount:rows.length,columnCount:count}).assign({
  margins:{top:data.compact?2:5,bottom:data.compact?2:5,left:8,right:8}
 });
 for(let i=0;i<rows.length;i++){
   t.rows[i].height=heights[i];
   for(let j=0;j<count;j++){
     const c=t.getCell(i,j);
     c.fill=i===0?C.pale:C.bg;
     c.text.style={typeface:'Lato',fontSize:font,bold:i===0,color:i===0?C.blue:C.ink,autoFit:'none',wrap:'none',insets:gap};
   }
 }
 return t;
}
function chart(slide,data,x,y,w,h){
 const d=data.chart;
 const isLine=d.type==='line';
 // Single-line group labels prevent native chart applications from suppressing
 // alternate category ticks when a long category wraps onto several lines.
 const displayCategories=data.kind==='importance'
  ? ['Âge véhicule','Contrat / usage','Technique / prix','Conducteur / bonus','Département','Marque/modèle/énergie/type']
  : d.categories;
 const cats=displayCategories.map(v=>wrap(v,isLine?130:235,18));
 if(CANVA_COMPATIBLE){
  // Import-only vector construction. Every label, bar, point and line remains
  // editable in Canva. Native chart series/workbooks remain in the final PPTX.
  const fmt=(v,n=d.decimals)=>v.toLocaleString('fr-FR',{minimumFractionDigits:n,maximumFractionDigits:n});
  const values=d.series[0].values;
  if(isLine){
   txt(slide,'Score public Kaggle',x,y,w,18,C.muted,true);
   const px=x+110,py=y+34,pw=w-170,ph=225;
   for(let k=0;k<=Math.floor((d.max-d.min)/.05);k++){
    const v=d.min+k*.05,gy=py+ph-ph*(v-d.min)/(d.max-d.min);
    line(slide,px,gy,pw,C.rule,.65);
    txt(slide,fmt(v,3),x,gy-12,94,17,C.muted,false,{align:'right',noWrap:true});
   }
   const points=values.map((v,i)=>({x:px+i*pw/(values.length-1),y:py+ph-(v-d.min)/(d.max-d.min)*ph}));
   for(let i=1;i<points.length;i++){
    const a=points[i-1],b=points[i];
    // Thin native rectangles are rotated around their own centers.
    const length=Math.hypot(b.x-a.x,b.y-a.y),angle=Math.atan2(b.y-a.y,b.x-a.x)*180/Math.PI;
    slide.shapes.add({geometry:'rect',position:{left:(a.x+b.x-length)/2,top:(a.y+b.y)/2-1.5,width:length,height:3,rotation:angle},fill:C.blue,line:{fill:'none',width:0}});
   }
   points.forEach((p,i)=>{
    slide.shapes.add({geometry:'ellipse',position:{left:p.x-4,top:p.y-4,width:8,height:8},fill:C.blue,line:{fill:'none',width:0}});
    txt(slide,fmt(values[i]),i===0?p.x+3:p.x-58,p.y+(i===0?-60:[1,4].includes(i)?7:-28),116,18,C.ink,true,{align:i===0?'left':'center',noWrap:true});
    txt(slide,wrap(displayCategories[i],125,17),p.x-62,py+ph+13,125,17,C.muted,false,{align:'center',noWrap:true});
   });
  }else{
   const labels=data.kind==='importance'
    ? ['Âge du véhicule','Contrat et usage','Technique et prix','Conducteur et bonus','Département','Marque, modèle,\nénergie et type']
    : d.categories;
   const max=data.kind==='importance'?2500:1800,step=data.kind==='importance'?500:200;
   const left=data.kind==='importance'?225:120,px=x+left,py=y+8,pw=w-left-72,ph=h-84,rh=ph/values.length;
   for(let t=0;t<=max;t+=step){
    const gx=px+pw*t/max;
    rect(slide,gx,py,.65,ph,C.rule);
    txt(slide,fmt(t,0),gx-28,py+ph+9,56,17,C.muted,false,{align:'center',noWrap:true});
   }
   for(let row=0;row<values.length;row++){
    const i=values.length-1-row,v=values[i],gy=py+row*rh;
    const label=labels[i],lines=label.split('\n').length;
    txt(slide,label,x,gy+(rh-lines*21)/2,left-12,18,C.ink,false,{align:'right',noWrap:true});
    rect(slide,px,gy+rh*.2,pw*v/max,rh*.6,C.blue);
    txt(slide,fmt(v),px+pw*v/max+5,gy+(rh-22)/2,72,18,C.ink,true,{noWrap:true});
   }
   txt(slide,d.axis,px,py+ph+43,pw,18,C.muted,true,{align:'center',noWrap:true});
  }
  return;
 }
 const ch=slide.charts.add(d.type,{
  position:{left:x,top:y,width:w,height:h},categories:cats,
  series:d.series.map((s,i)=>({...s,values:s.values.map(v=>Number(v.toFixed(d.decimals))),categories:cats,fill:i?C.green:C.blue,line:{style:'solid',fill:C.blue,width:3},marker:{symbol:isLine?'circle':'none',size:7,fill:C.blue}})),
  hasLegend:false,chartFill:C.bg,plotAreaFill:C.bg,
  barOptions:{direction:'bar',grouping:'clustered',gapWidth:80},
  lineOptions:{smooth:false},
  xAxis:isLine?{textStyle:{fontSize:18,fill:C.muted},line:{fill:C.rule,width:1}}:
       {textStyle:{fontSize:18,fill:C.ink},line:{fill:'none',width:0}},
  yAxis:isLine?{min:d.min,max:d.max,numberFormatCode:'0.000',majorGridlines:{fill:C.rule,width:.6},textStyle:{fontSize:18,fill:C.muted},title:{text:'Score public Kaggle',textStyle:{fontSize:18,fill:C.muted}}}:
       {min:0,textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:C.rule,width:.6},title:{text:d.axis,textStyle:{fontSize:18,fill:C.muted}}},
  dataLabels:{showValue:true,position:isLine?'top':'outEnd',numberFormatCode:d.decimals===0?'0':d.decimals===3?'0.000':'0.0',textStyle:{fontSize:isLine?20:18,fill:C.ink,bold:true}},
 });
 applyPresentationChartFont(ch,{fontFamily:'Lato'});
 return ch;
}
function boxText(slide,label,body,x,y,w,h,color=C.blue){
 const b=rect(slide,x,y,w,h,C.bg,color,1.5);
 txt(slide,label,x+16,y+15,w-32,23,color,true);
 txt(slide,body,x+16,y+53,w-32,24);
 return b;
}
function build(data){
 currentNumber=data.number;
 let slide;
 if(data.kind==='cover'){
  slide=P.slides.add();slide.background.fill=C.bg;
  txt(slide,'PROJET DE DATA SCIENCE · TARIFICATION AUTOMOBILE',64,54,1152,18,C.blue,true);
  txt(slide,'L’histoire de notre modèle\nde tarification',64,157,1130,64,C.ink,true);
  txt(slide,data.lead,68,355,1070,31,C.muted);
  line(slide,68,470,90,C.green,4);
  txt(slide,'Prime pure = probabilité de sinistre\n× montant moyen si sinistre',68,500,1110,32,C.blue,true);
  txt(slide,'Présentation complète : 25 à 30 minutes\n32 diapositives principales et 11 annexes',68,630,1000,21,C.muted);
  slide.speakerNotes.text=data.notes+'\n\nSOURCES\n'+data.refs.map(k=>model.sources[k].path??model.sources[k].url).join('\n');
  return;
 }
 slide=base(data);
 if(data.kind==='roadmap'){
  const stages=data.steps;
  const boxes=stages.map((s,i)=>boxText(slide,String(i+1).padStart(2,'0'),s,64+i*237,267,208,142));
  for(let i=0;i<4;i++)connect(slide,boxes[i],boxes[i+1]);
  const returns=data.returns;
  for(let i=0;i<3;i++){
   txt(slide,`RETOUR ${i+1}`,64,454+i*48,123,18,C.green,true);
   txt(slide,returns[i],205,450+i*48,1000,24,C.ink);
  }
 }else if(data.kind==='formula'||data.kind==='submission'){
  const size=data.formula.length>75?30:data.formula.length>60?33:39;
  txt(slide,data.formula,64,252,1152,size,C.blue,true);
  pairs(slide,data.left,64,344,536,270,24);
  pairs(slide,data.right,684,344,532,270,24);
 }else if(data.kind==='architecture'){
  const d=boxText(slide,'Contrat / prospect','Caractéristiques X',64,358,240,130,C.ink);
  const f=boxText(slide,'Fréquence p(X)','Logistique + correction\npar courbes',425,258,380,130,C.blue);
  const g=boxText(slide,'Gravité s(X)','0,375 Gamma + 0,375 Gamma-XGB\n+ 0,25 Gamma-Ridge',425,450,480,143,C.sev);
  const z=boxText(slide,'Prime pure','p(X) × s(X)',1010,358,206,130,C.ink);
  connect(slide,d,f);connect(slide,d,g,C.sev);
  connect(slide,f,z);connect(slide,g,z,C.sev);
 }else if(data.kind==='target'){
  if(CANVA_COMPATIBLE){
   txt(slide,'50 000 contrats d’apprentissage',64,269,550,25,C.ink,true);
   rect(slide,64,350,550,68,'#D8E2EA');
   rect(slide,64+550*(47083/50000),350,550*(2917/50000),68,C.blue);
   txt(slide,'Sans sinistre : 94,166 %',64,447,550,25,C.muted);
   txt(slide,'Avec sinistre : 5,834 %',64,497,550,25,C.blue,true);
  }else{
  const cc=slide.charts.add('doughnut',{
   position:{left:64,top:253,width:550,height:342},categories:['Sans sinistre','Sinistre'],
   series:[{name:'Contrats',values:[47083,2917],points:[{idx:0,fill:'#D8E2EA'},{idx:1,fill:C.blue}]}],
   doughnutOptions:{holeSize:67},hasLegend:false,
   dataLabels:{showPercent:true,numberFormatCode:'0.0%',textStyle:{fontSize:22,fill:C.ink}}
  });
  applyPresentationChartFont(cc,{fontFamily:'Lato'});
  }
  txt(slide,'Bleu : sinistres. Gris : absence de sinistre.',64,596,550,20,C.muted);
  pairs(slide,data.right,724,255,492,360,27);
 }else if(['chart','importance'].includes(data.kind)){
  chart(slide,data,64,247,760,365);
  pairs(slide,data.right,880,260,336,355,23);
 }else if(data.kind==='public_history'){
  chart(slide,data,64,249,1152,340);
  txt(slide,'Référence et v4 : registre historique. Autres scores : résultats utilisateur, non revérifiés. Axe vertical resserré.',64,600,1152,18,C.muted);
 }else if(data.kind==='correlations'){
  table(slide,data,data.rows,64,252,760,335);
  pairs(slide,data.left,880,257,336,330,24);
 }else if(data.kind==='table'){
  const h=data.rows.length>10?385:data.compact?365:340;
  table(slide,data,data.rows,64,data.rows.length>10?242:252,1152,h);
  if(data.left.length){
   const vals=data.left.map(v=>`${v[0]} : ${v[1]}`).join('\n');
   txt(slide,vals,64,585,1152,21,C.muted);
  }
 }else if(data.kind==='validation'){
  const a=boxText(slide,'Apprentissage autorisé','Imputations, catégories\nmodèles et auxiliaires',64,275,475,150);
  const b=boxText(slide,'Clients exclus','Prédire p et s\npuis mesurer la prime',742,275,474,150);
  connect(slide,a,b,C.green);
  txt(slide,'Un même client ne traverse jamais la frontière du pli',64,470,1152,31,C.green,true);
  txt(slide,'Au début : validation historique d’environ 20 %. Puis : plis internes et externes.\nDiagnostic final : recette fixe réapprise dans 3 plis sur le train complet.',64,534,1152,25);
 }else if(data.kind==='confusion'){
  for(let i=0;i<2;i++){
   const x=64+i*622;
   txt(slide,data.matrices[i].title,x,252,530,26,C.blue,true);
   const m=data.matrices[i].values;
   table(slide,{compact:false,number:data.number},[['Réalité / prédiction','0','1'],['Sans sinistre',m[0][0],m[0][1]],['Sinistre',m[1][0],m[1][1]]],x,315,530,220);
  }
  txt(slide,'AUC finale : 0,6533. Log-loss : 0,21393. Brier : 0,05407.',64,568,1152,26,C.muted);
 }else if(data.kind==='reference_image'){
  slide.images.add({blob:roadmapBytes,contentType:'image/png',fit:'contain',alt:'Roadmap fournie par l’utilisateur, avec cinq étapes et flèches de retour',position:{left:270,top:235,width:740,height:385}});
 }else{
  pairs(slide,data.left,64,253,536,355,25);
  pairs(slide,data.right,684,253,532,355,25);
 }
}

const roadmapBytes=new Uint8Array(await fs.readFile(path.join(OUT,'assets/roadmap_reference.png')));
for(const s of model.slides)build(s);
await fs.writeFile(path.join(BUILD,'text-layout-evidence.json'),JSON.stringify(textEvidence,null,2));
await fs.writeFile(path.join(BUILD,'presentation-proto.json'),JSON.stringify(P.toProto()));
const draft=path.join(BUILD,'candidate.pptx');
await(await PresentationFile.exportPptx(P)).save(draft);
execFileSync(path.join(RUNTIME,'python/bin/python3'),[path.join(ROOT,'scripts/repair_history_pptx.py'),draft],{stdio:'inherit'});
console.log('PPTX DRAFT',draft);
if(process.argv.includes('--quick')||CANVA_COMPATIBLE)process.exit(0);
const final=path.join(OUT,'histoire_modeles_tarification.pptx');
// Finalization uses a real local temporary filesystem: the shared checkout
// cannot apply the permissions used to freeze embedded chart workbooks.
const validationRoot=await fs.mkdtemp('/tmp/auto-pricing-presentation-');
await fs.mkdir(path.join(validationRoot,'validated'),{recursive:true});
await fs.mkdir(path.join(validationRoot,'validation'),{recursive:true});
const stagedCandidate=path.join(validationRoot,'candidate.pptx');
const stagedFinal=path.join(validationRoot,'validated/histoire_modeles_tarification.pptx');
const stagedReceipt=path.join(validationRoot,'validation/presentation.validation.json');
await fs.writeFile(stagedCandidate,await fs.readFile(draft));
await finalizePresentation({workspaceDir:validationRoot,candidatePath:stagedCandidate,finalPath:stagedFinal,
 pythonExecutable:path.join(RUNTIME,'python/bin/python3'),
 integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',
 ...model.slides.filter(s=>['table','correlations','confusion'].includes(s.kind)).flatMap(s=>['--require-native-table-slide',String(s.number)])],
 requiredNativeTableOwnerSlides:model.slides.filter(s=>['table','correlations','confusion'].includes(s.kind)).map(s=>s.number),
 requiredNativeChartOwnerSlides:model.slides.filter(s=>['target','chart','importance','public_history'].includes(s.kind)).map(s=>s.number),
 // The runtime's optional importer times out for this deck. Package, geometry,
 // fonts and native-chart validation still run; target-app review is separate.
 materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Lato']},verifyArtifactToolImport:false,
 receiptPath:stagedReceipt});
// Read/write preserves the verified bytes without copying filesystem modes.
await fs.writeFile(final,await fs.readFile(stagedFinal));
await fs.writeFile(path.join(BUILD,'presentation.validation.json'),await fs.readFile(stagedReceipt));
if(!process.argv.includes('--no-render')){
 // Render the actual, validated PPTX rather than the pre-export authoring model.
 execFileSync('libreoffice',[
  `-env:UserInstallation=${pathToFileURL(path.join(validationRoot,'lo-profile')).href}`,
  '--headless','--convert-to','pdf','--outdir',OUT,final
 ],{stdio:'inherit',timeout:120000});
 execFileSync('pdftoppm',[
  '-png','-scale-to-x','1280','-scale-to-y','720',
  path.join(OUT,'histoire_modeles_tarification.pdf'),path.join(BUILD,'previews/final')
 ],{stdio:'inherit',timeout:120000});
}
console.log('FINAL',final);
