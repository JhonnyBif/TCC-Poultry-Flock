import { Component, Inject } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MAT_DIALOG_DATA, MatDialogModule, MatDialogRef } from '@angular/material/dialog';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-ponto-selector-modal',
  standalone: true,
  imports: [MatButtonModule, MatIconModule, MatDialogModule,],
  templateUrl: './ponto-selector-modal.component.html',
  styleUrl: './ponto-selector-modal.component.scss',
})
export class PontoSelectorModalComponent {
  pontos: { x: number; y: number }[] = [];
  pontosFicticios: { x: number; y: number }[] = [];

  constructor(
    public dialogRef: MatDialogRef<PontoSelectorModalComponent>,
    @Inject(MAT_DIALOG_DATA) public data: { imagemBase64: string }
  ) {}

  registrarClique(event: MouseEvent) {
    if (this.pontos.length !== 4) {
      const imgElement = event.target as HTMLImageElement;

      const boundingRect = imgElement.getBoundingClientRect();

      const xClicado = event.clientX - boundingRect.left;
      const yClicado = event.clientY - boundingRect.top;

      this.pontosFicticios.push({ x: xClicado, y: yClicado });

      // Criar uma imagem temporária para obter as dimensões reais
      const tempImg = new Image();
      tempImg.style.display = 'none';
      tempImg.src = this.data.imagemBase64;

      tempImg.onload = () => {
        const escalaX = tempImg.width / imgElement.clientWidth;
        const escalaY = tempImg.height / imgElement.clientHeight;

        const xReal = Math.round(xClicado * escalaX);
        const yReal = Math.round(yClicado * escalaY);

        if (this.pontos.length < 4) {
          this.pontos.push({ x: xReal, y: yReal });
        }
      };
    }
  }

  enviar() {
    this.dialogRef.close(this.pontos);
  }
}
