import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { FilePreview } from '../interfaces/file-preview.interface';
import { apiUrlUtil } from '../../core/utils/api.utils';

@Injectable({
  providedIn: 'root',
})
export class UploadService {
  private apiUrlProcessar = `${apiUrlUtil}/processar`;
  private apiUrlTreinamento = `${apiUrlUtil}/treinamento`;

  constructor(private http: HttpClient) {}

  enviarVideo(
    video: FilePreview,
    selectedOption: string,
    parteCima: boolean,
    parteBaixo: boolean,
    regionPoints: { x: number; y: number }[] | null,
    medidas: {
      superior: string;
      esquerda: string;
      baixo: string;
      direita: string;
      dataVideo: string;
      idade: string;
      dataEntrada: string;
    } | null,
    datasets:any,
  ) {
    // console.log(dataset)
    const formData = new FormData();
    formData.append('file', video.file);
    formData.append('datasets',JSON.stringify(datasets) );
    formData.append('selectedOption', selectedOption);
    formData.append('parteCima', parteCima.toString());
    formData.append('parteBaixo', parteBaixo.toString());
    if (regionPoints) {
      formData.append('regionPoints', JSON.stringify(regionPoints));
    }
    if (medidas) {
      formData.append('superior', medidas.superior);
      formData.append('esquerda', medidas.esquerda);
      formData.append('baixo', medidas.baixo);
      formData.append('direita', medidas.direita);
      formData.append('idade', medidas.idade);
      formData.append('dataEntrada', medidas.dataEntrada);
      formData.append('dataVideo', medidas.dataVideo);
    }

    return this.http.post<any>(this.apiUrlProcessar, formData);
  }

  salvarImagemParaTreinamento(
    observacoesSalvas: {
      pontos: { x: number; y: number }[];
      texto: string;
      infos: string;
      data: string;
      idade: string;
      peso: string;
      image: string;
    }[]
  ) {
    return this.http.post<any>(this.apiUrlTreinamento, observacoesSalvas);
  }
}
